#!/usr/bin/env python3
"""
scripts/train_hierarchical_mamba.py

Orchestration runner for Stage 4: Hierarchical 3D Mamba.
Architecture:
  3D Patch Embedding -> Local Windowed Mamba (w=2) -> Token Reduction -> Global Mamba -> [mean, max] pooling -> MLP Head.

Executes:
  1. Overfit test on 8 volumes (confirms memorization).
  2. Single-fold sanity check on fold 0 (inspects individual patient predictions).
  3. Full 5-fold Cross-Validation with patient-level mean logit pooling.
  4. Decision threshold calibration (Youden-J optimal) alongside default tau=0.50.
  5. Compute logging and three-way comparative benchmark:
     Stage 2 (TinyCNN3D) vs Stage 3 (FlatMamba3D) vs Stage 4 (HierarchicalMamba3D).
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from datasets.volume_dataset import VolumeDataset
from models.hierarchical_mamba import HierarchicalMamba3D, get_model_param_count
from training.trainer import Trainer


def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    """Configures console and file logging."""
    logger = logging.getLogger("CancerMamba-Stage4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def collect_system_info(device: torch.device, model: nn.Module, backend: str) -> Dict[str, Any]:
    """Records hardware and environment details."""
    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stage": 4,
        "model_name": "HierarchicalMamba3D",
        "ssm_backend": backend,
        "window_size": model.window_size,
        "num_windows": model.num_windows,
        "token_reduction": model.token_reduction_method,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "device": str(device),
        "model_parameter_count": get_model_param_count(model),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = torch.cuda.get_device_capability(0)
        info["vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)
    return info


def build_model(config: Dict[str, Any]) -> HierarchicalMamba3D:
    """Builds HierarchicalMamba3D model from config dictionary."""
    return HierarchicalMamba3D(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
        d_model=config["model"]["d_model"],
        patch_size=config["model"]["patch_size"],
        volume_shape=tuple(config["data"]["input_shape"][1:]),
        window_size=config["model"].get("window_size", 2),
        local_layers=config["model"].get("local_layers", 2),
        global_layers=config["model"].get("global_layers", 2),
        token_reduction=config["model"].get("token_reduction", "learned_proj"),
        d_state=config["model"]["d_state"],
        expand=config["model"]["expand"],
        d_conv=config["model"]["d_conv"],
        dropout=config["model"]["dropout"],
    )


def compute_youden_metrics(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates per-fold optimal Youden-J metrics and aggregate statistics."""
    baccs, senss, specs, threshs = [], [], [], []

    for f in fold_results:
        tbl = f["final_patient_table"]
        y_true = np.array([p["true_label"] for p in tbl])
        y_prob = np.array([p["predicted_prob"] for p in tbl])

        best_j = -1.0
        best = {"t": 0.5, "bacc": 0.5, "sens": 0.0, "spec": 0.0}

        for t in np.linspace(0.01, 0.99, 100):
            preds = (y_prob >= t).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            tn = np.sum((preds == 0) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            j = sens + spec - 1.0

            if j > best_j:
                best_j = j
                best = {"t": t, "bacc": 0.5 * (sens + spec), "sens": sens, "spec": spec}

        baccs.append(best["bacc"])
        senss.append(best["sens"])
        specs.append(best["spec"])
        threshs.append(best["t"])

    return {
        "balanced_accuracy": {
            "mean": round(float(np.mean(baccs)), 4),
            "std": round(float(np.std(baccs)), 4),
            "per_fold": [round(float(v), 4) for v in baccs],
        },
        "sensitivity": {
            "mean": round(float(np.mean(senss)), 4),
            "std": round(float(np.std(senss)), 4),
            "per_fold": [round(float(v), 4) for v in senss],
        },
        "specificity": {
            "mean": round(float(np.mean(specs)), 4),
            "std": round(float(np.std(specs)), 4),
            "per_fold": [round(float(v), 4) for v in specs],
        },
        "threshold": {
            "mean": round(float(np.mean(threshs)), 4),
            "std": round(float(np.std(threshs)), 4),
            "per_fold": [round(float(v), 4) for v in threshs],
        },
    }


def run_single_fold_sanity(
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
    val_fold: int = 0,
    epochs: int = 5,
) -> Dict[str, Any]:
    """Runs single-fold check on fold 0 and displays visual patient prediction table."""
    logger.info(f"\n{'='*70}\n[SANITY CHECK 2/3] Single-Fold Sanity Check (Val Fold = {val_fold})\n{'='*70}")

    train_folds = [f for f in config["cv"]["folds"] if f != val_fold]
    train_dataset = VolumeDataset(
        patient_index_path=config["data"]["patient_index"],
        splits_path=config["data"]["splits"],
        folds=train_folds,
        verbose=False,
    )
    val_dataset = VolumeDataset(
        patient_index_path=config["data"]["patient_index"],
        splits_path=config["data"]["splits"],
        folds=[val_fold],
        verbose=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"] and device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    model = build_model(config)
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        pos_weight=config["training"].get("pos_weight"),
        use_amp=config["training"].get("amp", True),
        grad_accum_steps=config["training"].get("grad_accum_steps", 1),
        logger=logger,
    )

    fold_result = trainer.train_fold(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        fold_idx=val_fold,
    )

    logger.info("\nVisual Sanity Check Table (Validation Patients):")
    logger.info(f"{'Patient ID':16s} | {'True':4s} | {'Pred Prob':9s} | {'Pred Class':10s} | {'Volumes':7s} | Raw Logits")
    logger.info("-" * 75)
    for row in fold_result["final_patient_table"][:10]:
        logger.info(
            f"{row['patient_id']:16s} | {row['true_label']:4d} | {row['predicted_prob']:9.4f} | "
            f"{row['predicted_class']:10d} | {row['num_volumes']:7d} | {row['raw_logits']}"
        )

    return fold_result


def run_full_cross_validation(
    config: Dict[str, Any],
    device: torch.device,
    run_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Runs 5-fold cross-validation and computes primary and calibrated metrics."""
    logger.info(f"\n{'='*70}\n[RUNNING FULL 5-FOLD CROSS-VALIDATION (HIERARCHICAL MAMBA)]\n{'='*70}")

    all_folds = config["cv"]["folds"]
    fold_results = []
    epochs = config["training"]["epochs"]
    peak_vrams = []

    for val_fold in all_folds:
        train_folds = [f for f in all_folds if f != val_fold]

        train_ds = VolumeDataset(
            patient_index_path=config["data"]["patient_index"],
            splits_path=config["data"]["splits"],
            folds=train_folds,
            verbose=False,
        )
        val_ds = VolumeDataset(
            patient_index_path=config["data"]["patient_index"],
            splits_path=config["data"]["splits"],
            folds=[val_fold],
            verbose=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            pin_memory=config["data"]["pin_memory"] and device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
        )

        model = build_model(config)
        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            pos_weight=config["training"].get("pos_weight"),
            use_amp=config["training"].get("amp", True),
            grad_accum_steps=config["training"].get("grad_accum_steps", 1),
            logger=logger,
        )

        res = trainer.train_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            fold_idx=val_fold,
        )
        fold_results.append(res)
        peak_vrams.append(res["peak_vram_mb"])

    # Aggregate metrics across folds at default tau=0.50
    metric_keys = ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "sensitivity", "specificity"]
    pat_agg = {}
    vol_agg = {}

    for k in metric_keys:
        pat_vals = [f["best_patient_metrics"].get(k) for f in fold_results if f["best_patient_metrics"].get(k) is not None]
        if pat_vals:
            pat_agg[k] = {
                "mean": round(float(np.mean(pat_vals)), 4),
                "std": round(float(np.std(pat_vals)), 4),
                "per_fold": [round(float(v), 4) for v in pat_vals],
            }
        else:
            pat_agg[k] = {"mean": None, "std": None, "per_fold": []}

        vol_vals = [f["best_volume_metrics"].get(k) for f in fold_results if f["best_volume_metrics"].get(k) is not None]
        if vol_vals:
            vol_agg[k] = {
                "mean": round(float(np.mean(vol_vals)), 4),
                "std": round(float(np.std(vol_vals)), 4),
                "per_fold": [round(float(v), 4) for v in vol_vals],
            }
        else:
            vol_agg[k] = {"mean": None, "std": None, "per_fold": []}

    # Calibrated metrics via Youden-J threshold per fold
    youden_metrics = compute_youden_metrics(fold_results)

    max_vram_mb = max(peak_vrams) if peak_vrams else 0.0
    total_cv_time = sum(f["total_time_seconds"] for f in fold_results)

    logger.info(f"\n{'='*80}")
    logger.info("       STAGE 4: 5-FOLD CV SUMMARY (HIERARCHICAL MAMBA 3D)")
    logger.info(f"{'='*80}")
    logger.info(f"{'Metric':20s} | {'Patient-Level (Primary)':26s} | {'Volume-Level (Inflated)':26s}")
    logger.info("-" * 80)
    for k in metric_keys:
        p_str = f"{pat_agg[k]['mean']} +/- {pat_agg[k]['std']}" if pat_agg[k]['mean'] is not None else "N/A"
        v_str = f"{vol_agg[k]['mean']} +/- {vol_agg[k]['std']}" if vol_agg[k]['mean'] is not None else "N/A"
        logger.info(f"{k:20s} | {p_str:26s} | {v_str:26s}")
    logger.info("-" * 80)
    logger.info(f"Calibrated (Youden-J Threshold: {youden_metrics['threshold']['mean']} +/- {youden_metrics['threshold']['std']}):")
    logger.info(f"  Balanced Accuracy:  {youden_metrics['balanced_accuracy']['mean']} +/- {youden_metrics['balanced_accuracy']['std']}")
    logger.info(f"  Sensitivity:        {youden_metrics['sensitivity']['mean']} +/- {youden_metrics['sensitivity']['std']}")
    logger.info(f"  Specificity:        {youden_metrics['specificity']['mean']} +/- {youden_metrics['specificity']['std']}")
    logger.info("-" * 80)
    logger.info(f"Peak VRAM Usage:        {max_vram_mb:.2f} MB (out of 3712 MB budget, {max_vram_mb/3712*100:.1f}%)")
    logger.info(f"Total 5-Fold Run Time:  {total_cv_time:.1f} seconds (mean {total_cv_time/len(all_folds):.1f}s / fold)")
    logger.info(f"{'='*80}\n")

    summary = {
        "metadata": {
            "experiment": config["experiment"]["name"],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "epochs_per_fold": epochs,
            "n_folds": len(all_folds),
            "peak_vram_mb": max_vram_mb,
            "total_cv_time_seconds": round(total_cv_time, 2),
        },
        "aggregate_metrics": {
            "patient_level_primary": pat_agg,
            "volume_level_inflated": vol_agg,
            "patient_level_calibrated_youden": youden_metrics,
        },
        "fold_details": fold_results,
    }
    return summary


def three_way_comparative_benchmark(
    hier_summary: Dict[str, Any],
    cnn_metrics_path: Optional[Path],
    flat_mamba_path: Optional[Path],
    logger: logging.Logger,
):
    """Prints side-by-side three-way comparative benchmark under identical calibration."""
    if not cnn_metrics_path.exists() or not flat_mamba_path.exists():
        logger.info("Prior baseline metrics not found; skipping three-way table.")
        return

    with open(cnn_metrics_path, "r", encoding="utf-8") as f:
        cnn_data = json.load(f)
    with open(flat_mamba_path, "r", encoding="utf-8") as f:
        mamba_data = json.load(f)

    cnn_pat = cnn_data["cross_validation"]["aggregate_metrics"]["patient_level_primary"]
    flat_pat = mamba_data["cross_validation"]["aggregate_metrics"]["patient_level_primary"]
    hier_pat = hier_summary["aggregate_metrics"]["patient_level_primary"]

    cnn_meta = cnn_data["cross_validation"]["metadata"]
    flat_meta = mamba_data["cross_validation"]["metadata"]
    hier_meta = hier_summary["metadata"]

    logger.info(f"\n{'='*96}")
    logger.info("       THREE-WAY COMPARATIVE BENCHMARK: CNN vs FLAT MAMBA vs HIERARCHICAL MAMBA")
    logger.info(f"{'='*96}")
    logger.info(f"{'Metric':24s} | {'Stage 2 (TinyCNN3D)':21s} | {'Stage 3 (FlatMamba3D)':21s} | {'Stage 4 (Hierarchical)':21s}")
    logger.info("-" * 96)

    for k in ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "sensitivity", "specificity"]:
        c_val = f"{cnn_pat[k]['mean']} +/- {cnn_pat[k]['std']}"
        f_val = f"{flat_pat[k]['mean']} +/- {flat_pat[k]['std']}"
        h_val = f"{hier_pat[k]['mean']} +/- {hier_pat[k]['std']}"
        logger.info(f"{k:24s} | {c_val:21s} | {f_val:21s} | {h_val:21s}")

    logger.info("-" * 96)
    logger.info(f"{'Peak VRAM (MB)':24s} | {cnn_meta['peak_vram_mb']:<21.1f} | {flat_meta['peak_vram_mb']:<21.1f} | {hier_meta['peak_vram_mb']:<21.1f}")
    logger.info(f"{'Total CV Time (s)':24s} | {cnn_meta['total_cv_time_seconds']:<21.1f} | {flat_meta['total_cv_time_seconds']:<21.1f} | {hier_meta['total_cv_time_seconds']:<21.1f}")
    
    speedup = flat_meta['total_cv_time_seconds'] / max(hier_meta['total_cv_time_seconds'], 1e-4)
    logger.info(f"{'CV Speedup vs Flat':24s} | {'N/A':<21s} | {'1.0x (Reference)':<21s} | {f'{speedup:.2f}x faster':<21s}")
    logger.info(f"{'='*96}\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Hierarchical 3D Mamba Runner")
    parser.add_argument("--config", type=Path, default=Path("configs/hierarchical_mamba.yaml"))
    parser.add_argument("--mode", type=str, default="all", choices=["overfit", "single_fold", "full_cv", "all"])
    parser.add_argument("--cnn-metrics", type=Path, default=Path("runs/20260905_163731_baseline_cnn/metrics.json"))
    parser.add_argument("--flat-mamba-metrics", type=Path, default=Path("runs/20260906_073403_mamba3d_flat/metrics.json"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config["logging"]["runs_dir"]) / f"{timestamp}_hierarchical_mamba"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=run_dir / "training.log")
    logger.info(f"Created run directory: {run_dir}")
    logger.info(f"Target Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"SSM Backend: {config['experiment'].get('backend', 'pure_pytorch')}")

    ref_model = build_model(config)
    param_count = get_model_param_count(ref_model)
    logger.info(f"HierarchicalMamba3D Parameter Count: {param_count:,} params (<1M target)")
    logger.info(f"Patch Size: {ref_model.patch_size}, Window Size: {ref_model.window_size} -> Regional Windows: {ref_model.num_windows}")

    sys_info = collect_system_info(device, ref_model, config["experiment"].get("backend", "pure_pytorch"))
    with open(run_dir / "system_info.json", "w", encoding="utf-8") as f:
        json.dump(sys_info, f, indent=2)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    results_report = {"system_info": sys_info, "config": config}

    # 1. Overfit test
    if args.mode in ["overfit", "all"]:
        dataset_all = VolumeDataset(
            patient_index_path=config["data"]["patient_index"],
            splits_path=config["data"]["splits"],
            verbose=True,
        )
        overfit_config = dict(config)
        overfit_config["model"]["dropout"] = 0.0
        overfit_model = build_model(overfit_config)
        overfit_trainer = Trainer(
            model=overfit_model,
            device=device,
            learning_rate=config["overfit_test"]["learning_rate"],
            use_amp=config["training"].get("amp", True),
            logger=logger,
        )
        overfit_res = overfit_trainer.run_overfit_test(
            dataset=dataset_all,
            num_samples=config["overfit_test"]["num_samples"],
            epochs=config["overfit_test"]["epochs"],
            target_loss=config["overfit_test"]["threshold_loss"],
            target_acc=config["overfit_test"]["threshold_acc"],
        )
        results_report["overfit_test"] = overfit_res

    # 2. Single-fold sanity check
    if args.mode in ["single_fold", "all"]:
        single_res = run_single_fold_sanity(
            config=config,
            device=device,
            logger=logger,
            val_fold=config["sanity_run"]["val_fold"],
            epochs=config["sanity_run"]["epochs"],
        )
        results_report["single_fold_sanity"] = single_res

    # 3. Full 5-Fold Cross-Validation
    if args.mode in ["full_cv", "all"]:
        cv_summary = run_full_cross_validation(
            config=config,
            device=device,
            run_dir=run_dir,
            logger=logger,
        )
        results_report["cross_validation"] = cv_summary

        three_way_comparative_benchmark(cv_summary, args.cnn_metrics, args.flat_mamba_metrics, logger)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=2)

    logger.info(f"Stage 4 complete! All artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
