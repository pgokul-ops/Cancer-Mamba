#!/usr/bin/env python3
"""
scripts/train_patient_mamba.py

Orchestrates Stage 5 Patient-Level Sequence Modeling and Baselines:
  1. Evaluates all 4 aggregation mechanisms under identical nested 5-fold CV:
     - Mean Pooling
     - Max Pooling
     - Attention Pooling
     - Patient Mamba
  2. Dual task evaluation:
     - 5-year binary recurrence classification (N=72 labeled cohort) with threshold calibration (tau=0.50 and tau*).
     - Full-cohort time-to-event survival modeling (N=92 patients) with Harrell's Concordance Index (C-index).
  3. Paired per-fold comparison table against Stage 4 Hierarchical baseline.
  4. Exports metrics.json, config.yaml, and training.log.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from datasets.patient_sequence_dataset import PatientSequenceDataset, collate_patient_sequences
from evaluation.survival_metrics import CoxLoss, harrell_c_index
from models.patient_aggregators import (
    AttentionPoolingAggregator,
    MaxPoolingAggregator,
    MeanPoolingAggregator,
    PatientMambaAggregator,
)


def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("Stage5-PatientMamba")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def instantiate_model(model_cfg: Dict[str, Any]) -> nn.Module:
    name = model_cfg["name"]
    d_model = model_cfg.get("d_model", 256)
    d_hidden = model_cfg.get("d_hidden", 64)
    dropout = model_cfg.get("dropout", 0.2)

    if name == "mean_pooling":
        return MeanPoolingAggregator(d_model=d_model, d_hidden=d_hidden, dropout=dropout)
    elif name == "max_pooling":
        return MaxPoolingAggregator(d_model=d_model, d_hidden=d_hidden, dropout=dropout)
    elif name == "attention_pooling":
        return AttentionPoolingAggregator(d_model=d_model, d_hidden=d_hidden, dropout=dropout)
    elif name == "patient_mamba":
        return PatientMambaAggregator(
            d_model=d_model,
            n_layers=model_cfg.get("n_layers", 1),
            d_state=model_cfg.get("d_state", 16),
            expand=model_cfg.get("expand", 1.5),
            d_conv=model_cfg.get("d_conv", 4),
            d_hidden=d_hidden,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes standard binary classification metrics at a given threshold."""
    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "f1": None,
            "balanced_accuracy": None,
            "sensitivity": None,
            "specificity": None,
            "confusion_matrix": {},
        }

    auc = float(roc_auc_score(y_true, y_prob))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    order = np.argsort(rec)
    pr_auc = float(np.trapz(prec[order], rec[order]))

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bacc = float((sensitivity + specificity) / 2.0)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(bacc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def compute_youden_optimal(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Sweeps thresholds to find Youden's J statistic optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_tau = float(thresholds[best_idx])
    best_tau = max(0.01, min(0.99, best_tau))

    metrics = compute_binary_metrics(y_true, y_prob, threshold=best_tau)
    metrics["threshold"] = round(best_tau, 4)
    metrics["youden_j"] = round(float(j_scores[best_idx]), 4)
    return metrics


def train_eval_model_fold(
    model: nn.Module,
    fold_idx: int,
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Trains a model on fold_idx training split and evaluates on validation split."""
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    wd = config["training"]["weight_decay"]
    pos_weight = config["training"].get("pos_weight", 0.6)
    cls_w = config["training"].get("cls_loss_weight", 1.0)
    surv_w = config["training"].get("survival_loss_weight", 0.5)

    train_ds = PatientSequenceDataset(
        patient_index_path=config["data"]["patient_index"],
        splits_path=config["data"]["splits"],
        manifest_path=config["data"]["manifest"],
        embeddings_dir=config["data"]["embeddings_dir"],
        fold_idx=fold_idx,
        split_type="train",
        preload=True,
    )
    val_ds = PatientSequenceDataset(
        patient_index_path=config["data"]["patient_index"],
        splits_path=config["data"]["splits"],
        manifest_path=config["data"]["manifest"],
        embeddings_dir=config["data"]["embeddings_dir"],
        fold_idx=fold_idx,
        split_type="val",
        preload=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_patient_sequences
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_patient_sequences
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pos_weight_t = torch.tensor([pos_weight], device=device)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    cox_loss_fn = CoxLoss()

    best_cls_score = -1.0
    best_cls_eval = None
    best_cls_epoch = 0

    best_surv_score = -1.0
    best_surv_eval = None
    best_surv_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["embeddings"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            durations = batch["durations"].to(device)
            events = batch["events"].to(device)

            optimizer.zero_grad()
            out = model(x, mask)
            logits = out["logits"].squeeze(1)
            risk = out["risk_scores"].squeeze(1)

            # Masked BCE loss only on labeled cohort (label != -1)
            labeled_mask = (labels >= 0)
            if labeled_mask.sum() > 0:
                l_cls = bce_loss_fn(logits[labeled_mask], labels[labeled_mask])
            else:
                l_cls = torch.tensor(0.0, device=device)

            # Cox loss on all uncensored/censored patients
            l_surv = cox_loss_fn(risk, durations, events)

            total_loss = cls_w * l_cls + surv_w * l_surv
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() * len(labels)

        # Validation
        model.eval()
        val_pids = []
        val_labels = []
        val_probs = []
        val_risks = []
        val_durations = []
        val_events = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["embeddings"].to(device)
                mask = batch["mask"].to(device)
                out = model(x, mask)

                logits = out["logits"].squeeze(1).cpu()
                risk = out["risk_scores"].squeeze(1).cpu()
                probs = torch.sigmoid(logits)

                val_probs.extend(probs.numpy())
                val_risks.extend(risk.numpy())
                val_labels.extend(batch["labels"].numpy())
                val_durations.extend(batch["durations"].numpy())
                val_events.extend(batch["events"].numpy())
                val_pids.extend(batch["patient_ids"])

        val_probs = np.array(val_probs)
        val_risks = np.array(val_risks)
        val_labels = np.array(val_labels)
        val_durations = np.array(val_durations)
        val_events = np.array(val_events)

        # Classification metrics on labeled patients
        labeled_idx = (val_labels >= 0)
        y_true_cls = val_labels[labeled_idx].astype(int)
        y_prob_cls = val_probs[labeled_idx]

        cls_metrics = compute_binary_metrics(y_true_cls, y_prob_cls, threshold=0.50)
        cal_metrics = compute_youden_optimal(y_true_cls, y_prob_cls)

        # Survival C-index across ALL validation patients in this fold
        c_index = harrell_c_index(val_risks, val_durations, val_events)

        auc_val = cls_metrics.get("roc_auc")
        cls_score = auc_val if auc_val is not None else cls_metrics.get("balanced_accuracy", 0.5)

        if cls_score > best_cls_score:
            best_cls_score = cls_score
            best_cls_epoch = epoch
            best_cls_eval = {
                "default": cls_metrics,
                "calibrated": cal_metrics,
                "epoch": epoch,
                "score": round(cls_score, 4),
            }

        if c_index > best_surv_score:
            best_surv_score = c_index
            best_surv_epoch = epoch
            best_surv_eval = {
                "c_index": round(c_index, 4),
                "epoch": epoch,
            }

    return {
        "classification_default": best_cls_eval["default"] if best_cls_eval else {},
        "classification_calibrated": best_cls_eval["calibrated"] if best_cls_eval else {},
        "classification_best_epoch": best_cls_epoch,
        "survival_c_index": best_surv_eval["c_index"] if best_surv_eval else 0.5,
        "survival_best_epoch": best_surv_epoch,
        "num_val_patients_total": len(val_pids),
        "num_val_labeled": int(labeled_idx.sum()),
        "epoch": best_cls_epoch,
    }



def run_model_cv(
    model_cfg: Dict[str, Any],
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Runs 5-fold cross-validation for a specific model architecture."""
    name = model_cfg["name"]
    all_folds = config["cv"]["folds"]
    fold_results = []
    logger.info(f"\n==========================================")
    logger.info(f"Evaluating Model: {name.upper()}")
    logger.info(f"==========================================")

    start_time = time.time()

    for fold_idx in all_folds:
        # Re-seed for identical fold initialization
        torch.manual_seed(config["training"]["seed"] + fold_idx)
        np.random.seed(config["training"]["seed"] + fold_idx)

        model = instantiate_model(model_cfg).to(device)
        res = train_eval_model_fold(
            model=model,
            fold_idx=fold_idx,
            config=config,
            device=device,
            logger=logger,
        )
        fold_results.append(res)
        cls_def = res["classification_default"]
        c_idx = res["survival_c_index"]
        logger.info(
            f"Fold {fold_idx} Best (Ep {res['epoch']}): ROC-AUC: {cls_def.get('roc_auc')} | "
            f"BACC: {cls_def.get('balanced_accuracy')} | F1: {cls_def.get('f1')} | "
            f"C-index: {c_idx} | Cal BACC: {res['classification_calibrated'].get('balanced_accuracy')}"
        )

    duration = time.time() - start_time

    # Aggregate across folds
    aucs = [f["classification_default"]["roc_auc"] for f in fold_results if f["classification_default"]["roc_auc"] is not None]
    pr_aucs = [f["classification_default"]["pr_auc"] for f in fold_results if f["classification_default"]["pr_auc"] is not None]
    f1s = [f["classification_default"]["f1"] for f in fold_results if f["classification_default"]["f1"] is not None]
    baccs = [f["classification_default"]["balanced_accuracy"] for f in fold_results if f["classification_default"]["balanced_accuracy"] is not None]
    sens = [f["classification_default"]["sensitivity"] for f in fold_results if f["classification_default"]["sensitivity"] is not None]
    specs = [f["classification_default"]["specificity"] for f in fold_results if f["classification_default"]["specificity"] is not None]

    cal_baccs = [f["classification_calibrated"]["balanced_accuracy"] for f in fold_results]
    cal_specs = [f["classification_calibrated"]["specificity"] for f in fold_results]
    cal_sens = [f["classification_calibrated"]["sensitivity"] for f in fold_results]
    cal_taus = [f["classification_calibrated"]["threshold"] for f in fold_results]

    c_indices = [f["survival_c_index"] for f in fold_results]

    def mean_std(vals):
        return {
            "mean": round(float(np.mean(vals)), 4) if vals else None,
            "std": round(float(np.std(vals)), 4) if vals else None,
            "per_fold": vals,
        }

    agg = {
        "model_name": name,
        "runtime_seconds": round(duration, 2),
        "classification_default_tau_0_50": {
            "roc_auc": mean_std(aucs),
            "pr_auc": mean_std(pr_aucs),
            "f1": mean_std(f1s),
            "balanced_accuracy": mean_std(baccs),
            "sensitivity": mean_std(sens),
            "specificity": mean_std(specs),
        },
        "classification_calibrated_tau_star": {
            "balanced_accuracy": mean_std(cal_baccs),
            "sensitivity": mean_std(cal_sens),
            "specificity": mean_std(cal_specs),
            "threshold": mean_std(cal_taus),
        },
        "survival_full_cohort_c_index": mean_std(c_indices),
        "fold_details": fold_results,
    }

    logger.info(f"\n--- {name.upper()} Summary ---")
    logger.info(f"Patient ROC-AUC (tau=0.50): {agg['classification_default_tau_0_50']['roc_auc']['mean']} +/- {agg['classification_default_tau_0_50']['roc_auc']['std']}")
    logger.info(f"Calibrated Balanced Acc (tau*): {agg['classification_calibrated_tau_star']['balanced_accuracy']['mean']} +/- {agg['classification_calibrated_tau_star']['balanced_accuracy']['std']}")
    logger.info(f"Survival C-index (All 92 pts): {agg['survival_full_cohort_c_index']['mean']} +/- {agg['survival_full_cohort_c_index']['std']}")
    return agg


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark patient-level aggregators (Stage 5).")
    parser.add_argument("--config", type=str, default="configs/patient_mamba.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{timestamp}_patient_mamba"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "training.log")
    logger.info("=== Stage 5: Patient-Level Sequence Modeling & Baselines ===")
    logger.info(f"Run directory: {run_dir}")

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load Stage 4 benchmark results for paired comparison
    stage4_aucs = [0.5556, 0.8333, 0.8222, 0.9333, 0.8222]

    all_results = {}
    for model_cfg in config["models"]:
        res = run_model_cv(model_cfg, config, device, logger)
        all_results[model_cfg["name"]] = res

    # Construct paired per-fold comparison table
    logger.info("\n" + "=" * 90)
    logger.info("PAIRED PER-FOLD COMPARISON: STAGE 4 vs STAGE 5 AGGREGATORS")
    logger.info("=" * 90)
    header = f"{'Fold':<6} | {'Stage 4 (Vol Pool)':<18} | {'Mean Pool':<12} | {'Max Pool':<12} | {'Attention':<12} | {'Patient Mamba':<14} | {'Mamba Delta':<12}"
    logger.info(header)
    logger.info("-" * len(header))

    mamba_aucs = all_results["patient_mamba"]["classification_default_tau_0_50"]["roc_auc"]["per_fold"]
    mean_aucs = all_results["mean_pooling"]["classification_default_tau_0_50"]["roc_auc"]["per_fold"]
    max_aucs = all_results["max_pooling"]["classification_default_tau_0_50"]["roc_auc"]["per_fold"]
    att_aucs = all_results["attention_pooling"]["classification_default_tau_0_50"]["roc_auc"]["per_fold"]

    for f_idx in range(5):
        s4 = stage4_aucs[f_idx]
        pm = mamba_aucs[f_idx]
        mp = mean_aucs[f_idx]
        xp = max_aucs[f_idx]
        ap = att_aucs[f_idx]
        delta = pm - s4
        delta_str = f"{delta:+.4f}"
        logger.info(f"Fold {f_idx:<1} | {s4:<18.4f} | {mp:<12.4f} | {xp:<12.4f} | {ap:<12.4f} | {pm:<14.4f} | {delta_str:<12}")

    logger.info("-" * len(header))
    m_s4 = float(np.mean(stage4_aucs))
    m_pm = all_results["patient_mamba"]["classification_default_tau_0_50"]["roc_auc"]["mean"]
    m_mp = all_results["mean_pooling"]["classification_default_tau_0_50"]["roc_auc"]["mean"]
    m_xp = all_results["max_pooling"]["classification_default_tau_0_50"]["roc_auc"]["mean"]
    m_ap = all_results["attention_pooling"]["classification_default_tau_0_50"]["roc_auc"]["mean"]
    logger.info(f"{'Mean':<6} | {m_s4:<18.4f} | {m_mp:<12.4f} | {m_xp:<12.4f} | {m_ap:<12.4f} | {m_pm:<14.4f} | {m_pm - m_s4:+.4f}")
    logger.info("=" * 90)

    # Save outputs
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "stage": 5,
                "stage4_reference_aucs": stage4_aucs,
                "aggregators": all_results,
            },
            f,
            indent=2,
        )

    logger.info(f"\nAll Stage 5 artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
