import os
import torch
import numpy as np
import sklearn.metrics as metrics
import mlflow
import pickle
import tempfile

from tqdm import tqdm


def compute_binary_metrics(labels, predictions, beta=1.0):
    precision = metrics.precision_score(labels, predictions, zero_division=0)
    recall = metrics.recall_score(labels, predictions, zero_division=0)
    f1 = metrics.f1_score(labels, predictions, zero_division=0)
    fbeta = metrics.fbeta_score(labels, predictions, beta=beta, zero_division=0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fbeta": fbeta,
    }


def score_metrics(metric_values, selection_metric="f1", selection_beta=1.0):
    if selection_metric == "precision":
        return metric_values["precision"]
    if selection_metric == "fbeta":
        if "fbeta" in metric_values:
            return metric_values["fbeta"]
        precision = metric_values["precision"]
        recall = metric_values["recall"]
        if precision == 0.0 and recall == 0.0:
            return 0.0
        beta_sq = selection_beta ** 2
        denominator = beta_sq * precision + recall
        if denominator == 0.0:
            return 0.0
        return (1 + beta_sq) * precision * recall / denominator
    return metric_values["f1"]


def select_best_threshold(
    labels,
    probabilities,
    threshold_step=0.05,
    selection_metric="f1",
    selection_beta=1.0,
):
    if threshold_step <= 0 or threshold_step > 1:
        raise ValueError(f"threshold_step must be in (0, 1], got {threshold_step}")

    best_threshold = 0.5
    best_score = -1.0
    best_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "fbeta": 0.0,
    }

    num_steps = int(round(1.0 / threshold_step))
    candidate_thresholds = [round(i * threshold_step, 10) for i in range(num_steps + 1)]
    if candidate_thresholds[-1] != 1.0:
        candidate_thresholds.append(1.0)

    for threshold in candidate_thresholds:
        predictions = [1 if prob > threshold else 0 for prob in probabilities]
        current_metrics = compute_binary_metrics(labels, predictions, beta=selection_beta)

        current_score = score_metrics(
            current_metrics,
            selection_metric=selection_metric,
            selection_beta=selection_beta,
        )

        # Break ties toward higher precision, then higher recall, to reduce over-matching.
        if (
            current_score > best_score
            or (
                np.isclose(current_score, best_score)
                and current_metrics["precision"] > best_metrics["precision"]
            )
            or (
                np.isclose(current_score, best_score)
                and np.isclose(current_metrics["precision"], best_metrics["precision"])
                and current_metrics["recall"] > best_metrics["recall"]
            )
        ):
            best_score = current_score
            best_threshold = float(threshold)
            best_metrics = current_metrics

    return best_metrics, best_threshold


def blocked_matmul(mata, matb,
                   threshold=None,
                   k=None,
                   batch_size=512):
    """Find the most similar pairs of vectors from two matrices (top-k or threshold)

    Args:
        mata (np.ndarray): the first matrix
        matb (np.ndarray): the second matrix
        threshold (float, optional): if set, return all pairs of cosine
            similarity above the threshold
        k (int, optional): if set, return for each row in matb the top-k
            most similar vectors in mata
        batch_size (int, optional): the batch size of each block

    Returns:
        list of tuples: the pairs of similar vectors' indices and the similarity
    """
    mata = np.array(mata)
    matb = np.array(matb)
    results = []
    for start in tqdm(range(0, len(matb), batch_size)):
        block = matb[start:start+batch_size]
        sim_mat = np.matmul(mata, block.transpose())
        if k is not None:
            indices = np.argpartition(-sim_mat, k, axis=0)
            for row in indices[:k]:
                for idx_b, idx_a in enumerate(row):
                    idx_b += start
                    results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))
        elif threshold is not None:
            indices = np.argwhere(sim_mat >= threshold)
            for idx_a, idx_b in indices:
                idx_b += start
                results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))
    return results


def evaluate(
    model,
    iterator,
    threshold=None,
    threshold_step=0.05,
    selection_metric="f1",
    selection_beta=1.0,
):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            if model.task_type == 'em':
                x1, x2, x12, y = batch
                logits = model(x1, x2, x12)
            else:
                x, y = batch
                logits = model(x)

            # print(probs)
            probs = logits.softmax(dim=1)[:, 1]

            # print(logits)
            # pred = logits.argmax(dim=1)
            all_probs += probs.cpu().numpy().tolist()
            # all_p += pred.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            pickle.dump(pred, temp_file)
            temp_path = temp_file.name
        try:
            mlflow.log_artifact(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        current_metrics = compute_binary_metrics(all_y, pred, beta=selection_beta)
        return current_metrics["f1"], current_metrics["precision"], current_metrics["recall"]

    best_metrics, best_th = select_best_threshold(
        all_y,
        all_probs,
        threshold_step=threshold_step,
        selection_metric=selection_metric,
        selection_beta=selection_beta,
    )
    return best_metrics["f1"], best_metrics["precision"], best_metrics["recall"], best_th
