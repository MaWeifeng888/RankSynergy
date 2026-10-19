"""Shared reproducibility and regression metrics for research experiments."""

import random

import numpy as np
from scipy.stats import rankdata


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _validated_pair(labels, preds):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.shape != preds.shape:
        raise ValueError("Labels and predictions must have equal, nonempty shapes.")
    if not np.isfinite(labels).all() or not np.isfinite(preds).all():
        raise ValueError("Labels and predictions must contain only finite values.")
    return labels, preds


def safe_corr(labels, preds, kind="pearson"):
    labels, preds = _validated_pair(labels, preds)
    if kind not in {"pearson", "spearman"}:
        raise ValueError(f"Unsupported correlation: {kind}")
    if labels.size < 2 or np.ptp(labels) == 0 or np.ptp(preds) == 0:
        return 0.0
    if kind == "spearman":
        labels = rankdata(labels, method="average")
        preds = rankdata(preds, method="average")
    return float(np.corrcoef(labels, preds)[0, 1])


def calc_metrics(labels, preds):
    labels, preds = _validated_pair(labels, preds)
    residual_sum = np.sum((labels - preds) ** 2)
    total_sum = np.sum((labels - labels.mean()) ** 2)
    mse = float(residual_sum / labels.size)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "scc": safe_corr(labels, preds, "spearman"),
        "pcc": safe_corr(labels, preds),
        "r2": float(1.0 - residual_sum / total_sum) if total_sum > 0 else 0.0,
    }
