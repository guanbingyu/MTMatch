import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse
import mlflow

from .utils import evaluate, score_metrics
from .model import DMModel
from .dataset import DMDataset

from torch.utils import data
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from apex import amp


def train_step(train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    class_weights = torch.tensor(
        [hp.neg_class_weight, hp.pos_class_weight],
        dtype=torch.float,
        device=model.device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        if len(batch) == 4:
            x1, x2, x12, y = batch
            prediction = model(x1, x2, x12)
        else:
            x, y = batch
            prediction = model(x)

        loss = criterion(prediction, y.to(model.device))
        # loss = criterion(prediction, y.float().to(model.device))
        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss


def train(trainset, validset, testset, run_tag, hp):
    """Train and evaluate the model

    Args:
        trainset (DMDataset): the training set
        validset (DMDataset): the validation set
        testset (DMDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Ditto
    model = DMModel(device=device, lm=hp.lm, task_type=hp.task_type, 
                    pretrained=(hp.ssl_method == 'mtl'))

    model = model.cuda()
    start_epoch = 0
    best_dev_f1 = best_test_f1 = 0.0
    best_dev_score = -1.0
    if hp.resume_from:
        saved_state = torch.load(hp.resume_from, map_location='cpu')
        state_dict = saved_state["model"] if isinstance(saved_state, dict) and "model" in saved_state else saved_state
        model.load_state_dict(state_dict, strict=False)
        start_epoch = int(saved_state.get("epoch", 0)) if isinstance(saved_state, dict) else 0
        metrics_blob = saved_state.get("metrics", {}) if isinstance(saved_state, dict) else {}
        if metrics_blob:
            best_dev_f1 = float(metrics_blob.get("dev_f1", 0.0))
            best_test_f1 = float(metrics_blob.get("test_f1", 0.0))
            best_dev_score = score_metrics(
                {
                    "precision": float(metrics_blob.get("dev_p", 0.0)),
                    "recall": float(metrics_blob.get("dev_r", 0.0)),
                    "f1": best_dev_f1,
                },
                selection_metric=hp.selection_metric,
                selection_beta=hp.selection_beta,
            )
        print(f"Resuming from {hp.resume_from} at epoch {start_epoch}")
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    remaining_epochs = max(0, hp.n_epochs - start_epoch)
    num_steps = (len(trainset) // hp.batch_size) * remaining_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)
    os.makedirs(hp.logdir, exist_ok=True)
    best_ckpt_path = os.path.join(hp.logdir, "best_model.pt")
    epoch_ckpt_dir = os.path.join(hp.logdir, "epoch_checkpoints")
    os.makedirs(epoch_ckpt_dir, exist_ok=True)
    if start_epoch >= hp.n_epochs:
        print(f"Resume checkpoint is already at epoch {start_epoch}, which reaches/exceeds --n_epochs={hp.n_epochs}. No further training run.")
        writer.close()
        return
    for epoch in range(start_epoch + 1, hp.n_epochs+1):
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp)

        # eval
        model.eval()
        dev_f1, dev_p, dev_r, th = evaluate(
            model,
            valid_iter,
            threshold_step=hp.threshold_step,
            selection_metric=hp.selection_metric,
            selection_beta=hp.selection_beta,
        )
        test_f1, test_p, test_r = evaluate(
            model,
            test_iter,
            threshold=th,
            selection_beta=hp.selection_beta,
        )

        current_dev_metrics = {
            "precision": dev_p,
            "recall": dev_r,
            "f1": dev_f1,
        }
        current_dev_score = score_metrics(
            current_dev_metrics,
            selection_metric=hp.selection_metric,
            selection_beta=hp.selection_beta,
        )

        if current_dev_score > best_dev_score:
            best_dev_score = current_dev_score
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "threshold": th,
                    "selection_metric": hp.selection_metric,
                    "selection_beta": hp.selection_beta,
                    "threshold_step": hp.threshold_step,
                    "metrics": {
                        "dev_f1": dev_f1,
                        "dev_p": dev_p,
                        "dev_r": dev_r,
                        "test_f1": test_f1,
                        "test_p": test_p,
                        "test_r": test_r,
                    },
                },
                best_ckpt_path,
            )
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "threshold": th,
                "selection_metric": hp.selection_metric,
                "selection_beta": hp.selection_beta,
                "threshold_step": hp.threshold_step,
                "metrics": {
                    "dev_f1": dev_f1,
                    "dev_p": dev_p,
                    "dev_r": dev_r,
                    "test_f1": test_f1,
                    "test_p": test_p,
                    "test_r": test_r,
                },
            },
            os.path.join(epoch_ckpt_dir, f"epoch_{epoch:03d}.pt"),
        )
        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   'p': dev_p,
                   'r': dev_r,
                   't_f1': test_f1,
                   't_p': test_p,
                   't_r': test_r}
        writer.add_scalars(run_tag, scalars, epoch)
        for variable in ["dev_f1", "dev_p", "dev_r", "test_f1", "test_p", "test_r"]:
            mlflow.log_metric(variable, eval(variable))


    writer.close()
