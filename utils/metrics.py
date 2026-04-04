import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Compute all classification metrics from raw logits and integer labels.

    Returns dict with: accuracy, auroc, f1, precision, recall.
    """
    probs  = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    preds  = logits.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "auroc":     float(roc_auc_score(labels, probs)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
    }


def compute_confusion_matrix(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """Return 2×2 confusion matrix as numpy array."""
    preds  = logits.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return confusion_matrix(labels, preds)
