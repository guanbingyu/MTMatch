import argparse
import os
import random

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.utils import data
from transformers import AutoModel

from selfsl.dataset import DMDataset
from selfsl.path_utils import resolve_task_dir
from selfsl.utils import compute_binary_metrics, select_best_threshold


LM_MAP = {
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    if os.path.isfile(checkpoint_path):
        parent_dir = os.path.dirname(checkpoint_path)
        preferred_sibling = os.path.join(parent_dir, "best_model.pt")
        if os.path.isfile(preferred_sibling):
            return preferred_sibling
        return checkpoint_path

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    candidates = [
        os.path.join(checkpoint_path, "best_model.pt"),
        os.path.join(checkpoint_path, "ssl.pt"),
        os.path.join(checkpoint_path, "mtmatch_pretrain_data", "ssl.pt"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"No supported checkpoint found under directory: {checkpoint_path}"
    )


def evaluate_with_threshold(
    model,
    iterator,
    threshold=None,
    threshold_step=0.05,
    selection_metric="f1",
    selection_beta=1.0,
):
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in iterator:
            if model.task_type == "em":
                x1, x2, x12, labels = batch
                logits = model(x1, x2, x12)
            else:
                x, labels = batch
                logits = model(x)

            probs = logits.softmax(dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    if threshold is None:
        best_metrics, best_threshold = select_best_threshold(
            all_labels,
            all_probs,
            threshold_step=threshold_step,
            selection_metric=selection_metric,
            selection_beta=selection_beta,
        )
        return best_metrics["f1"], best_metrics["precision"], best_metrics["recall"], best_threshold

    pred = [1 if prob > threshold else 0 for prob in all_probs]
    current_metrics = compute_binary_metrics(all_labels, pred, beta=selection_beta)
    f1 = current_metrics["f1"]
    precision = current_metrics["precision"]
    recall = current_metrics["recall"]
    return f1, precision, recall


def load_state_dict_from_checkpoint(model, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)


class AdaptiveCheckpointModel(nn.Module):
    def __init__(self, lm: str, state_dict: dict, task_type: str):
        super().__init__()
        lm_name = LM_MAP.get(lm, lm)
        self.bert = AutoModel.from_pretrained(lm_name)
        self.task_type = task_type

        self.projector = None
        if "projector.weight" in state_dict and "projector.bias" in state_dict:
            out_dim, in_dim = state_dict["projector.weight"].shape
            self.projector = nn.Linear(in_dim, out_dim)

        if "fc.weight" not in state_dict:
            raise RuntimeError("Checkpoint missing fc.weight; cannot build evaluation head.")
        fc_out_dim, fc_in_dim = state_dict["fc.weight"].shape
        self.fc = nn.Linear(fc_in_dim, fc_out_dim)

    def _encode(self, x):
        rep = self.bert(x)[0][:, 0, :]
        if self.projector is not None:
            rep = self.projector(rep)
        return rep

    def forward(self, x1, x2=None, x12=None):
        if self.task_type == "em":
            if x12 is None:
                raise RuntimeError("EM mode requires x12 inputs.")

            x1 = x1.to(self.fc.weight.device)
            x2 = x2.to(self.fc.weight.device)
            x12 = x12.to(self.fc.weight.device)

            fc_in_dim = self.fc.weight.shape[1]
            pair_rep = self._encode(x12)

            if fc_in_dim == pair_rep.shape[1] * 2:
                batch_size = len(x1)
                enc = self._encode(torch.cat((x1, x2), dim=0))
                enc1 = enc[:batch_size]
                enc2 = enc[batch_size:]
                feat = torch.cat((pair_rep, (enc1 - enc2).abs()), dim=1)
                return self.fc(feat)

            if fc_in_dim == pair_rep.shape[1]:
                return self.fc(pair_rep)

            raise RuntimeError(
                f"Unsupported fc input dim {fc_in_dim} for encoded dim {pair_rep.shape[1]}"
            )

        x1 = x1.to(self.fc.weight.device)
        rep = self._encode(x1)
        return self.fc(rep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Abt-Buy")
    parser.add_argument("--task_type", type=str, default="em")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt/.bin)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--size", type=int, default=None, help="Optional subset size for quick evaluation")
    parser.add_argument("--lm", type=str, default="roberta")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="f1",
        choices=["f1", "fbeta", "precision"],
        help="Validation objective used for threshold selection.",
    )
    parser.add_argument(
        "--selection_beta",
        type=float,
        default=1.0,
        help="Beta used when selection_metric=fbeta.",
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.05,
        help="Search step for threshold tuning on the validation set.",
    )

    args = parser.parse_args()

    args.checkpoint = resolve_checkpoint_path(args.checkpoint)

    set_seed(args.run_id)

    task_path = resolve_task_dir(args.task_type, args.task)
    valid_path = os.path.join(task_path, "valid.txt")
    test_path = os.path.join(task_path, "test.txt")

    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"Valid file not found: {valid_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    validset = DMDataset(valid_path, lm=args.lm, size=args.size, max_len=args.max_len)
    testset = DMDataset(test_path, lm=args.lm, size=None, max_len=args.max_len)

    valid_iter = data.DataLoader(
        dataset=validset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=validset.pad,
    )
    test_iter = data.DataLoader(
        dataset=testset,
        batch_size=max(1, args.batch_size * 8),
        shuffle=False,
        num_workers=0,
        collate_fn=testset.pad,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    model = AdaptiveCheckpointModel(lm=args.lm, state_dict=state_dict, task_type=args.task_type)
    load_state_dict_from_checkpoint(model, args.checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    valid_f1, valid_p, valid_r, best_threshold = evaluate_with_threshold(
        model,
        valid_iter,
        threshold=None,
        threshold_step=args.threshold_step,
        selection_metric=args.selection_metric,
        selection_beta=args.selection_beta,
    )
    test_f1, test_p, test_r = evaluate_with_threshold(
        model,
        test_iter,
        threshold=best_threshold,
        selection_beta=args.selection_beta,
    )

    print(f"task={args.task_type}/{args.task}")
    print(f"checkpoint={args.checkpoint}")
    print(f"valid: f1={valid_f1:.4f}, p={valid_p:.4f}, r={valid_r:.4f}, threshold={best_threshold:.2f}")
    print(f"test : f1={test_f1:.4f}, p={test_p:.4f}, r={test_r:.4f}")
