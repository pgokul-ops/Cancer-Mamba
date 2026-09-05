"""
evaluation/metrics.py

Patient-level and volume-level classification metric computation for Cancer-Mamba.
Implements:
  - Patient-level aggregation: groups per-volume logits by patient_id and takes mean
    before sigmoid thresholding.
  - Volume-level metrics (un-aggregated, for explicit comparison against inflated scores).
  - Metrics: ROC-AUC, PR-AUC, F1, Balanced Accuracy, Sensitivity, Specificity.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
import torch


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes standard evaluation metrics from ground-truth binary labels and predicted probabilities.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    num_samples = len(y_true)
    num_pos = int(np.sum(y_true == 1))
    num_neg = int(np.sum(y_true == 0))

    # ROC-AUC & PR-AUC require both positive and negative classes
    if num_pos > 0 and num_neg > 0:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = float("nan")

        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float("nan")
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    # F1 & Balanced Accuracy
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bacc = float(balanced_accuracy_score(y_true, y_pred))

    # Sensitivity (Recall) & Specificity from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
        "pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else None,
        "f1": round(f1, 4),
        "balanced_accuracy": round(bacc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "num_samples": num_samples,
        "num_positive": num_pos,
        "num_negative": num_neg,
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
    }


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluates model predictions both at the patient level (primary) and volume level (comparison).

    Args:
        predictions: List of dicts, each containing:
            'patient_id': str
            'study_series_id': str
            'logit': float or float-tensor
            'label': int or float
    Returns:
        Dict containing 'patient_level' and 'volume_level_inflated' metric dicts, plus patient prediction rows.
    """
    if not predictions:
        return {"patient_level": {}, "volume_level_inflated": {}, "patient_table": []}

    # 1. Volume-level (raw, unaggregated)
    vol_y_true = []
    vol_y_prob = []

    # 2. Patient-level aggregation: collect logits per patient
    patient_logits = defaultdict(list)
    patient_labels = {}

    for p in predictions:
        pid = p["patient_id"]
        logit = float(p["logit"]) if not isinstance(p["logit"], torch.Tensor) else float(p["logit"].item())
        label = int(p["label"])

        prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        vol_y_true.append(label)
        vol_y_prob.append(prob)

        patient_logits[pid].append(logit)
        if pid not in patient_labels:
            patient_labels[pid] = label

    # Compute volume-level metrics
    vol_metrics = compute_binary_classification_metrics(
        np.array(vol_y_true), np.array(vol_y_prob), threshold=threshold
    )

    # Compute patient-level metrics via mean-logit pooling
    pat_y_true = []
    pat_y_prob = []
    patient_table = []

    for pid in sorted(patient_logits.keys()):
        true_label = patient_labels[pid]
        mean_logit = float(np.mean(patient_logits[pid]))
        mean_prob = float(1.0 / (1.0 + np.exp(-mean_logit)))

        pat_y_true.append(true_label)
        pat_y_prob.append(mean_prob)

        patient_table.append({
            "patient_id": pid,
            "true_label": true_label,
            "predicted_prob": round(mean_prob, 4),
            "predicted_class": int(mean_prob >= threshold),
            "num_volumes": len(patient_logits[pid]),
            "raw_logits": [round(l, 3) for l in patient_logits[pid]],
        })

    pat_metrics = compute_binary_classification_metrics(
        np.array(pat_y_true), np.array(pat_y_prob), threshold=threshold
    )

    return {
        "patient_level": pat_metrics,
        "volume_level_inflated": vol_metrics,
        "patient_table": patient_table,
    }
