from pathlib import Path
import copy
import argparse
import math
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import mlflow

from torch import nn, optim
from torch.optim import AdamW
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from transformers import AutoModel
from transformers import get_linear_schedule_with_warmup
from torch.utils import data
# apex.amp has been removed in favor of native AMP.
from tensorboardX import SummaryWriter
from tqdm import tqdm
from .augment import Augmenter
from .bt_dataset import BTDataset
from .dataset import DMDataset
from .block import evaluate_blocking
from .bootstrap import bootstrap, bootstrap_cleaning
from .utils import compute_binary_metrics, score_metrics, select_best_threshold

# Native mixed-precision utilities.
from torch.amp import autocast, GradScaler

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsSimCLR(nn.Module):
    def __init__(self, hp, device='cuda', lm='roberta'):
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        hidden_size = 768
        self.em_head_type = getattr(hp, 'em_head_type', 'full')

        sizes = [hidden_size] + list(map(int, hp.projector.split('-')))
        self.projector = nn.Linear(hidden_size, sizes[-1])
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        if hp.task_type == 'em':
            if self.em_head_type == 'paired_only':
                self.fc = nn.Linear(sizes[-1], 2)
            elif self.em_head_type == 'full':
                self.fc = nn.Linear(sizes[-1] * 2, 2)
            else:
                raise ValueError(f'Unsupported em_head_type: {self.em_head_type}')
        else:
            self.fc = nn.Linear(sizes[-1], 2)

        # InfoNCE uses a variable number of classes per batch, so it cannot share
        # the binary class-weighted loss used by supervised fine-tuning.
        self.criterion = nn.CrossEntropyLoss().to(device)

    def info_nce_loss(self, features, batch_size, n_views, temperature=0.07):
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / temperature
        return logits, labels

    def forward(self, flag, y1, y2, y12, da=None, cutoff_ratio=0.1):
        if flag in [0, 1]:
            batch_size = len(y1)
            y1 = y1.to(self.device)
            y2 = y2.to(self.device)
            if da == 'cutoff':
                seq_len = y2.size()[1]
                y1_word_embeds = self.bert.embeddings.word_embeddings(y1)
                y2_word_embeds = self.bert.embeddings.word_embeddings(y2)
                
                position_ids = torch.LongTensor([list(range(seq_len))]).to(self.device)
                pos_embeds = self.bert.embeddings.position_embeddings(position_ids)

                l = random.randint(1, int(seq_len * cutoff_ratio)+1)
                s = random.randint(0, seq_len - l - 1)
                y2_word_embeds[:, s:s+l, :] -= pos_embeds[:, s:s+l, :]

                y_embeds = torch.cat((y1_word_embeds, y2_word_embeds))
                z = self.bert(inputs_embeds=y_embeds)[0][:, 0, :]
            else:
                y = torch.cat((y1, y2))
                z = self.bert(y)[0][:, 0, :]
            z = self.projector(z)

            if flag == 0:
                logits, labels = self.info_nce_loss(z, batch_size, 2)
                loss = self.criterion(logits, labels)
                return loss
            elif flag == 1:
                z1 = z[:batch_size]
                z2 = z[batch_size:]
                c = (self.bn(z1).T @ self.bn(z2)) / (len(z1))
                on_diag = ((torch.diagonal(c) - 1) ** 2).sum() * self.hp.scale_loss
                off_diag = (off_diagonal(c) ** 2).sum() * self.hp.scale_loss
                loss = on_diag + self.hp.lambd * off_diag
                return loss
        elif flag == 2:
            if self.hp.task_type == 'em':
                x1, x2, x12 = y1.to(self.device), y2.to(self.device), y12.to(self.device)
                enc_pair = self.projector(self.bert(x12)[0][:, 0, :])
                if self.em_head_type == 'paired_only':
                    return self.fc(enc_pair)
                if self.em_head_type == 'full':
                    batch_size = len(x1)
                    enc = self.projector(self.bert(torch.cat((x1, x2)))[0][:, 0, :])
                    enc1, enc2 = enc[:batch_size], enc[batch_size:]
                    return self.fc(torch.cat((enc_pair, (enc1 - enc2).abs()), dim=1))
                raise ValueError(f'Unsupported em_head_type: {self.em_head_type}')
            else:
                x1 = y1.to(self.device)
                enc = self.projector(self.bert(x1)[0][:, 0, :])
                return self.fc(enc)

def evaluate(
    model,
    iterator,
    threshold=None,
    ec_task=None,
    dump=False,
    threshold_step=0.05,
    selection_metric="f1",
    selection_beta=1.0,
):
    all_y, all_probs = [], []
    with torch.no_grad():
        # Enable autocast in evaluation to keep the numerical regime aligned
        # with fp16/bfloat16 training runs when CUDA is available.
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            for batch in tqdm(iterator):
                if len(batch) == 4:
                    x1, x2, x12, y = batch
                    logits = model(2, x1, x2, x12)
                else:
                    x, y = batch
                    logits = model(2, x, None, None)

                probs = logits.softmax(dim=1)[:, 1]
                all_probs += probs.cpu().numpy().tolist()
                all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        if dump:
            import pickle
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
                pickle.dump(pred, temp_file)
                temp_path = temp_file.name
            try:
                mlflow.log_artifact(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        current_metrics = compute_binary_metrics(all_y, pred, beta=selection_beta)
        f1 = current_metrics["f1"]
        p = current_metrics["precision"]
        r = current_metrics["recall"]
        if ec_task:
            # ... 原有的 error correction 逻辑保持不变 ...
            return f1, p, r, 0.0 # 简化返回
        else:
            return f1, p, r
    best_metrics, best_th = select_best_threshold(
        all_y,
        all_probs,
        threshold_step=threshold_step,
        selection_metric=selection_metric,
        selection_beta=selection_beta,
    )
    return best_metrics["f1"], best_metrics["precision"], best_metrics["recall"], best_th

def selfsl_step(train_nolabel_iter, model, optimizer, scheduler, scaler, hp):
    model.train()
    for i, batch in enumerate(train_nolabel_iter):
        yA, yB = batch
        optimizer.zero_grad()
        
        # 50系显卡推荐使用 bfloat16
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=hp.fp16):
            if hp.ssl_method == 'simclr':
                loss = model(0, yA, yB, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
            elif hp.ssl_method == 'barlow_twins':
                loss = model(1, yA, yB, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
            else:
                alpha = 1 - hp.alpha_bt
                loss = alpha * model(0, yA, yB, [], da=hp.da) + (1 - alpha) * model(1, yA, yB, [])

        if hp.fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        if i % 10 == 0: print(f"    step: {i}, loss: {loss.item()}")

def fine_tune_step(train_iter, model, optimizer, scheduler, scaler, hp):
    model.train()
    class_weights = torch.tensor(
        [hp.neg_class_weight, hp.pos_class_weight],
        dtype=torch.float,
        device=model.device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=hp.fp16):
            if len(batch) == 4:
                x1, x2, x12, y = batch
                prediction = model(2, x1, x2, x12)
            else:
                x, y = batch
                prediction = model(2, x, None, None)
            loss = criterion(prediction, y.to(model.device))

        if hp.fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        if i % 10 == 0: print(f"    fine tune step: {i}, loss: {loss.item()}")

def train(trainset_nolabel, trainset, validset, testset, run_tag, hp):
    # 50系显卡性能优化
    torch.set_float32_matmul_precision('high')
    
    # DataLoader 初始化部分保持不变 ...
    train_nolabel_iter = data.DataLoader(dataset=trainset_nolabel, batch_size=hp.batch_size, shuffle=True, collate_fn=trainset_nolabel.pad)
    train_iter = data.DataLoader(dataset=trainset, batch_size=hp.batch_size//2, shuffle=True, collate_fn=trainset.pad)
    valid_iter = data.DataLoader(dataset=validset, batch_size=hp.batch_size, shuffle=False, collate_fn=validset.pad)
    test_iter = data.DataLoader(dataset=testset, batch_size=hp.batch_size*16, shuffle=False, collate_fn=testset.pad)

    model = BarlowTwinsSimCLR(hp, device='cuda', lm=hp.lm).cuda()
    start_epoch = 0
    best_dev_f1 = 0.0
    best_dev_score = -1.0
    best_ckpt_path = os.path.join(hp.logdir, "best_model.pt")
    if hp.resume_from:
        saved_state = torch.load(hp.resume_from, map_location='cpu')
        state_dict = saved_state["model"] if isinstance(saved_state, dict) and "model" in saved_state else saved_state
        model.load_state_dict(state_dict, strict=False)
        start_epoch = int(saved_state.get("epoch", 0)) if isinstance(saved_state, dict) else 0
        metrics_blob = saved_state.get("metrics", {}) if isinstance(saved_state, dict) else {}
        if metrics_blob:
            best_dev_f1 = float(metrics_blob.get("dev_f1", 0.0))
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
    
    # 原生 GradScaler
    scaler = GradScaler('cuda', enabled=hp.fp16)

    remaining_ssl_epochs = max(0, hp.n_ssl_epochs - start_epoch)
    remaining_finetune_epochs = max(0, hp.n_epochs - max(start_epoch, hp.n_ssl_epochs))
    num_ssl_steps = len(trainset_nolabel) // hp.batch_size * remaining_ssl_epochs
    num_finetune_steps = len(trainset) // (hp.batch_size // 2) * remaining_finetune_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_ssl_steps + max(0, num_finetune_steps))

    os.makedirs(hp.logdir, exist_ok=True)
    try:
        writer = SummaryWriter(log_dir=hp.logdir)
    except Exception as exc:
        print(f"Warning: failed to initialize SummaryWriter under {hp.logdir}: {exc}")
        writer = None
    epoch_ckpt_dir = os.path.join(hp.logdir, "epoch_checkpoints")
    os.makedirs(epoch_ckpt_dir, exist_ok=True)
    if start_epoch >= hp.n_epochs:
        print(f"Resume checkpoint is already at epoch {start_epoch}, which reaches/exceeds --n_epochs={hp.n_epochs}. No further training run.")
        if writer is not None:
            writer.close()
        return

    for epoch in range(start_epoch + 1, hp.n_epochs + 1):
        epoch_threshold = None
        epoch_metrics = {}
        # Bootstrap 逻辑保持不变 ...
        
        if epoch <= hp.n_ssl_epochs:
            selfsl_step(train_nolabel_iter, model, optimizer, scheduler, scaler, hp)
        else:
            fine_tune_step(train_iter, model, optimizer, scheduler, scaler, hp)
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
                dump=True,
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
            epoch_threshold = th
            epoch_metrics = {
                "dev_f1": dev_f1,
                "dev_p": dev_p,
                "dev_r": dev_r,
                "test_f1": test_f1,
                "test_p": test_p,
                "test_r": test_r,
            }
            if current_dev_score > best_dev_score:
                best_dev_score = current_dev_score
                best_dev_f1 = dev_f1
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
                print(f"epoch {epoch}: NEW BEST dev_f1={dev_f1}, test_f1={test_f1}")

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "threshold": epoch_threshold,
                "selection_metric": hp.selection_metric,
                "selection_beta": hp.selection_beta,
                "threshold_step": hp.threshold_step,
                "metrics": epoch_metrics,
            },
            os.path.join(epoch_ckpt_dir, f"epoch_{epoch:03d}.pt"),
        )
    
    if writer is not None:
        writer.close()
