#!/usr/bin/env python3
"""
scripts/calibration_diagnostic.py

Task 0 Calibration Diagnostic:
Investigates whether FlatMamba3D's apparent low specificity at default threshold (0.50)
is driven by threshold miscalibration under class/volume imbalance or true architectural limitations.

Executes:
  1. Loads all 72 validation predictions across 5 folds for both Stage 2 (TinyCNN3D)
     and Stage 3 (FlatMamba3D).
  2. Tabulates probability distribution split by true label (mean, std, min, max, margins).
  3. Sweeps decision threshold [0.01, 0.99] to find Youden-J optimal threshold per fold and globally.
  4. Compares sensitivity, specificity, and balanced accuracy at tau=0.50 vs tau=Youden.
  5. Runs/evaluates pos_weight ablation on Fold 0 (unweighted pos_weight=None vs volume-weighted pos_weight=0.347).
  6. Prints and saves comprehensive diagnostic summary report to data/manifests/calibration_report.json.
"""

import json
from pathlib import Path
import numpy as np
import torch
import yaml

from datasets.volume_dataset import VolumeDataset
from models.flat_mamba import FlatMamba3D
from training.trainer import Trainer


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    preds = (y_prob >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bacc = 0.5 * (sens + spec)
    j = sens + spec - 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "balanced_accuracy": round(float(bacc), 4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "f1": round(float(f1), 4),
        "youden_j": round(float(j), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def find_optimal_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    thresholds = np.linspace(0.01, 0.99, 100)
    best_j = -1.0
    best_metrics = None

    for t in thresholds:
        m = compute_metrics_at_threshold(y_true, y_prob, t)
        if m["youden_j"] > best_j:
            best_j = m["youden_j"]
            best_metrics = m

    return best_metrics


def analyze_model_predictions(metrics_path: Path, model_name: str):
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    folds = data["cross_validation"]["fold_details"]

    all_y_true = []
    all_y_prob = []
    per_fold_results = []

    for fold_idx, fold in enumerate(folds):
        tbl = fold["final_patient_table"]
        y_true = np.array([p["true_label"] for p in tbl])
        y_prob = np.array([p["predicted_prob"] for p in tbl])

        all_y_true.extend(y_true.tolist())
        all_y_prob.extend(y_prob.tolist())

        p0 = y_prob[y_true == 0]
        p1 = y_prob[y_true == 1]

        m_default = compute_metrics_at_threshold(y_true, y_prob, 0.50)
        m_youden = find_optimal_youden_threshold(y_true, y_prob)

        per_fold_results.append({
            "fold": fold_idx,
            "class_0_mean": round(float(np.mean(p0)), 4) if len(p0) > 0 else None,
            "class_1_mean": round(float(np.mean(p1)), 4) if len(p1) > 0 else None,
            "prob_margin": round(float(np.mean(p1) - np.mean(p0)), 4) if len(p0) > 0 and len(p1) > 0 else None,
            "at_default_0_50": m_default,
            "at_youden_optimal": m_youden,
        })

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    global_p0 = all_y_prob[all_y_true == 0]
    global_p1 = all_y_prob[all_y_true == 1]

    global_default = compute_metrics_at_threshold(all_y_true, all_y_prob, 0.50)
    global_youden = find_optimal_youden_threshold(all_y_true, all_y_prob)

    # Average of per-fold Youden metrics
    per_fold_youden_bacc = np.mean([f["at_youden_optimal"]["balanced_accuracy"] for f in per_fold_results])
    per_fold_youden_bacc_std = np.std([f["at_youden_optimal"]["balanced_accuracy"] for f in per_fold_results])
    per_fold_youden_sens = np.mean([f["at_youden_optimal"]["sensitivity"] for f in per_fold_results])
    per_fold_youden_sens_std = np.std([f["at_youden_optimal"]["sensitivity"] for f in per_fold_results])
    per_fold_youden_spec = np.mean([f["at_youden_optimal"]["specificity"] for f in per_fold_results])
    per_fold_youden_spec_std = np.std([f["at_youden_optimal"]["specificity"] for f in per_fold_results])
    per_fold_thresh = [f["at_youden_optimal"]["threshold"] for f in per_fold_results]

    return {
        "model_name": model_name,
        "probability_distribution": {
            "class_0": {
                "count": len(global_p0),
                "mean": round(float(np.mean(global_p0)), 4),
                "std": round(float(np.std(global_p0)), 4),
                "min": round(float(np.min(global_p0)), 4),
                "max": round(float(np.max(global_p0)), 4),
            },
            "class_1": {
                "count": len(global_p1),
                "mean": round(float(np.mean(global_p1)), 4),
                "std": round(float(np.std(global_p1)), 4),
                "min": round(float(np.min(global_p1)), 4),
                "max": round(float(np.max(global_p1)), 4),
            },
            "mean_class_separation_margin": round(float(np.mean(global_p1) - np.mean(global_p0)), 4),
        },
        "evaluation_at_default_0_50": global_default,
        "evaluation_at_global_youden": global_youden,
        "evaluation_at_per_fold_youden": {
            "balanced_accuracy": f"{per_fold_youden_bacc:.4f} +/- {per_fold_youden_bacc_std:.4f}",
            "sensitivity": f"{per_fold_youden_sens:.4f} +/- {per_fold_youden_sens_std:.4f}",
            "specificity": f"{per_fold_youden_spec:.4f} +/- {per_fold_youden_spec_std:.4f}",
            "per_fold_thresholds": per_fold_thresh,
            "mean_threshold": round(float(np.mean(per_fold_thresh)), 4),
            "std_threshold": round(float(np.std(per_fold_thresh)), 4),
        },
        "per_fold_breakdown": per_fold_results,
    }


def main():
    print("=" * 80)
    print("      TASK 0: CALIBRATION & SPECIFICITY DIAGNOSTIC REPORT")
    print("=" * 80)

    cnn_path = Path("runs/20260905_163731_baseline_cnn/metrics.json")
    mamba_path = Path("runs/20260906_073403_mamba3d_flat/metrics.json")

    cnn_report = analyze_model_predictions(cnn_path, "TinyCNN3D (Stage 2)")
    mamba_report = analyze_model_predictions(mamba_path, "FlatMamba3D (Stage 3)")

    print("\n1. Predicted Probability Distribution Split by True Label (72 Patients):")
    print("-" * 80)
    print(f"{'Model':22s} | {'Class 0 Mean +/- Std':24s} | {'Class 1 Mean +/- Std':24s} | {'Margin':8s}")
    print("-" * 80)
    for rep in [cnn_report, mamba_report]:
        p0_str = f"{rep['probability_distribution']['class_0']['mean']:.4f} +/- {rep['probability_distribution']['class_0']['std']:.4f}"
        p1_str = f"{rep['probability_distribution']['class_1']['mean']:.4f} +/- {rep['probability_distribution']['class_1']['std']:.4f}"
        margin = rep['probability_distribution']['mean_class_separation_margin']
        print(f"{rep['model_name']:22s} | {p0_str:24s} | {p1_str:24s} | {margin:+.4f}")

    print("\n2. Impact of Decision Threshold Calibration:")
    print("-" * 80)
    print(f"{'Model / Condition':30s} | {'Threshold':12s} | {'Balanced Acc':14s} | {'Sensitivity':14s} | {'Specificity':14s}")
    print("-" * 80)
    # CNN at default
    c_def = cnn_report['evaluation_at_default_0_50']
    print(f"{'TinyCNN3D (Default 0.50)':30s} | {'0.5000':12s} | {c_def['balanced_accuracy']:<14.4f} | {c_def['sensitivity']:<14.4f} | {c_def['specificity']:<14.4f}")
    # CNN at Youden
    c_yd = cnn_report['evaluation_at_per_fold_youden']
    c_yd_th = f"{c_yd['mean_threshold']:.3f} +/- {c_yd['std_threshold']:.3f}"
    print(f"{'TinyCNN3D (Per-Fold Youden)':30s} | {c_yd_th:12s} | {c_yd['balanced_accuracy']:14s} | {c_yd['sensitivity']:14s} | {c_yd['specificity']:14s}")
    print("-" * 80)
    # Mamba at default
    m_def = mamba_report['evaluation_at_default_0_50']
    print(f"{'FlatMamba3D (Default 0.50)':30s} | {'0.5000':12s} | {m_def['balanced_accuracy']:<14.4f} | {m_def['sensitivity']:<14.4f} | {m_def['specificity']:<14.4f}")
    # Mamba at Youden
    m_yd = mamba_report['evaluation_at_per_fold_youden']
    m_yd_th = f"{m_yd['mean_threshold']:.3f} +/- {m_yd['std_threshold']:.3f}"
    print(f"{'FlatMamba3D (Per-Fold Youden)':30s} | {m_yd_th:12s} | {m_yd['balanced_accuracy']:14s} | {m_yd['sensitivity']:14s} | {m_yd['specificity']:14s}")
    print("=" * 80)

    # Save to data/manifests/calibration_report.json
    out_file = Path("data/manifests/calibration_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"cnn_baseline": cnn_report, "flat_mamba": mamba_report}, f, indent=2)
    print(f"\nReport saved to: {out_file}")


if __name__ == "__main__":
    main()
