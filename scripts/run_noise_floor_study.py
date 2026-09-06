#!/usr/bin/env python3
"""
scripts/run_noise_floor_study.py

STAGE 5.75 — ESTABLISH THE NOISE FLOOR (Gate Before Stage 6)
Project: Cancer-Mamba

Executes:
  TASK 1: Multi-seed variance check
    - Trains Stage 4 Hierarchical 3D Mamba across 3 seeds (42, 123, 456) under identical 5-fold CV.
    - Tracks validation metrics at every epoch (1..15) without using them for checkpoint selection.
    - Computes 5-fold mean ROC-AUC across seeds under:
        (a) Fixed Epoch 15 (Primary Honest Non-Peeking Criterion)
        (b) Fixed Epoch 10 (Intermediate Honest Criterion)
        (c) Min Train Loss (Training-Only Criterion)
        (d) Validation Peeking Best Epoch (Historical Rule)
    - Measures the empirical noise floor: Spread (max - min) and Std across seeds.

  TASK 2: Non-peeking baseline establishment
    - Fixes checkpoint selection to Fixed Epoch 15 for the reference model (Seed 42).
    - Saves checkpoints to runs/stage4_fixed_checkpoints/best_model_fold_{0..4}.pt.
    - Evaluates Stage 4 volume logit mean vs embedding mean then head under zero peeking.
    - Explicitly quantifies the validation-peeking inflation gap.

  TASK 3: Single Stage 5 comparison under the corrected protocol
    - Extracts pre-classifier volume embeddings using the exact Seed 42 Fixed Epoch 15 checkpoints.
    - Evaluates all 4 aggregators (Mean, Max, Attention, Patient Mamba) under identical warm-start.
    - Tests whether Patient Mamba's advantage clears the Task 1 noise floor.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "scripts"))

from datasets.volume_dataset import VolumeDataset
from models.hierarchical_mamba import HierarchicalMamba3D
from training.trainer import Trainer
from extract_embeddings import extract_fold_embeddings
from train_patient_mamba import (
    compute_stage4_control,
    instantiate_model,
    run_model_cv,
)


def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("Stage5.75-NoiseFloor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_hierarchical_model(config: Dict[str, Any]) -> HierarchicalMamba3D:
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


def train_and_track_fold(
    seed: int,
    fold_idx: int,
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
    is_reference_seed: bool = False,
    ref_ckpt_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Trains a fold model for the specified number of epochs, recording metrics at every epoch.
    Tracks fixed epoch 15, fixed epoch 10, min train loss, and val peeking.
    """
    all_folds = config["cv"]["folds"]
    train_folds = [f for f in all_folds if f != fold_idx]

    train_ds = VolumeDataset(
        patient_index_path=config["data"]["patient_index"],
        splits_path=config["data"]["splits"],
        folds=train_folds,
        verbose=False,
    )
    val_ds = VolumeDataset(
        patient_index_path=config["data"]["patient_index"],
        splits_path=config["data"]["splits"],
        folds=[fold_idx],
        verbose=False,
    )

    num_workers = config["data"].get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config["data"]["pin_memory"] and device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_hierarchical_model(config)
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

    epochs = config["training"]["epochs"]
    epoch_records = []

    best_val_auc = -1.0
    best_val_epoch = 0
    min_train_loss = float("inf")
    min_train_loss_epoch = 0

    state_dict_ep15 = None
    state_dict_best_val = None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, throughput, ep_dur = trainer.train_epoch(train_loader, epoch)
        eval_result = trainer.evaluate(val_loader)
        pat_auc = eval_result["patient_level"].get("roc_auc")
        vol_auc = eval_result["volume_level_inflated"].get("roc_auc")
        bacc = eval_result["patient_level"].get("balanced_accuracy")

        epoch_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_pat_auc": round(pat_auc, 4) if pat_auc is not None else None,
            "val_vol_auc": round(vol_auc, 4) if vol_auc is not None else None,
            "val_bacc": round(bacc, 4) if bacc is not None else None,
        })

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            min_train_loss_epoch = epoch

        score = pat_auc if pat_auc is not None else bacc
        if score > best_val_auc:
            best_val_auc = score
            best_val_epoch = epoch
            if is_reference_seed:
                state_dict_best_val = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}

        if epoch == epochs and is_reference_seed:
            state_dict_ep15 = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}

    total_time = time.time() - t0

    # Save reference checkpoints if reference seed
    if is_reference_seed and ref_ckpt_dir is not None:
        ref_ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_path = ref_ckpt_dir / f"best_model_fold_{fold_idx}.pt"
        torch.save(
            {
                "epoch": 15,
                "criterion": "fixed_epoch_15",
                "state_dict": state_dict_ep15,
                "best_score": float(rec_ep15.get("val_pat_auc", 0.0) or 0.0),
                "fold": fold_idx,
                "seed": seed,
            },
            save_path,
        )
        logger.info(f"  [Ref Seed {seed}] Saved Fold {fold_idx} Fixed Epoch 15 checkpoint to {save_path}")

    # Extract metrics under each criterion
    # 1. Fixed Epoch 15
    rec_ep15 = epoch_records[14] if len(epoch_records) >= 15 else epoch_records[-1]
    # 2. Fixed Epoch 10
    rec_ep10 = epoch_records[9] if len(epoch_records) >= 10 else epoch_records[-1]
    # 3. Min Train Loss
    rec_min_train = epoch_records[min_train_loss_epoch - 1]
    # 4. Validation Peeking Best Epoch
    rec_val_peek = epoch_records[best_val_epoch - 1]

    return {
        "fold": fold_idx,
        "seed": seed,
        "total_time_seconds": round(total_time, 2),
        "fixed_epoch_15": rec_ep15,
        "fixed_epoch_10": rec_ep10,
        "min_train_loss": {
            "best_epoch": min_train_loss_epoch,
            "metrics": rec_min_train,
        },
        "val_peeking": {
            "best_epoch": best_val_epoch,
            "metrics": rec_val_peek,
        },
        "all_epochs": epoch_records,
    }


def run_seed_cv(
    seed: int,
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
    is_reference_seed: bool = False,
    ref_ckpt_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Runs 5-fold cross-validation for a specific random seed."""
    logger.info(f"\n==================================================")
    logger.info(f"--- RUNNING 5-FOLD CV: SEED {seed} {'(REFERENCE)' if is_reference_seed else ''} ---")
    logger.info(f"==================================================")

    set_seed(seed)
    all_folds = config["cv"]["folds"]
    fold_results = []

    for f_idx in all_folds:
        # Per-fold seed
        set_seed(seed + f_idx)
        logger.info(f"Starting Seed {seed} | Fold {f_idx}...")
        res = train_and_track_fold(
            seed=seed,
            fold_idx=f_idx,
            config=config,
            device=device,
            logger=logger,
            is_reference_seed=is_reference_seed,
            ref_ckpt_dir=ref_ckpt_dir,
        )
        logger.info(
            f"Seed {seed} Fold {f_idx} Done ({res['total_time_seconds']:.1f}s) | "
            f"Fixed Ep15 AUC: {res['fixed_epoch_15']['val_pat_auc']} | "
            f"Fixed Ep10 AUC: {res['fixed_epoch_10']['val_pat_auc']} | "
            f"Val-Peek AUC: {res['val_peeking']['metrics']['val_pat_auc']} (ep {res['val_peeking']['best_epoch']})"
        )
        fold_results.append(res)

    # Compute 5-fold summary for each criterion
    def extract_aucs(extractor):
        return [extractor(f) for f in fold_results if extractor(f) is not None]

    aucs_ep15 = extract_aucs(lambda f: f["fixed_epoch_15"]["val_pat_auc"])
    aucs_ep10 = extract_aucs(lambda f: f["fixed_epoch_10"]["val_pat_auc"])
    aucs_min_train = extract_aucs(lambda f: f["min_train_loss"]["metrics"]["val_pat_auc"])
    aucs_peek = extract_aucs(lambda f: f["val_peeking"]["metrics"]["val_pat_auc"])

    summary = {
        "seed": seed,
        "fixed_epoch_15": {
            "mean": round(float(np.mean(aucs_ep15)), 4),
            "std": round(float(np.std(aucs_ep15)), 4),
            "per_fold": aucs_ep15,
        },
        "fixed_epoch_10": {
            "mean": round(float(np.mean(aucs_ep10)), 4),
            "std": round(float(np.std(aucs_ep10)), 4),
            "per_fold": aucs_ep10,
        },
        "min_train_loss": {
            "mean": round(float(np.mean(aucs_min_train)), 4),
            "std": round(float(np.std(aucs_min_train)), 4),
            "per_fold": aucs_min_train,
        },
        "val_peeking": {
            "mean": round(float(np.mean(aucs_peek)), 4),
            "std": round(float(np.std(aucs_peek)), 4),
            "per_fold": aucs_peek,
        },
        "peeking_inflation": round(float(np.mean(aucs_peek) - np.mean(aucs_ep15)), 4),
        "fold_details": fold_results,
    }

    logger.info(f"\n--- Seed {seed} 5-Fold Summary ---")
    logger.info(f"  Fixed Epoch 15 (Honest):  {summary['fixed_epoch_15']['mean']} +/- {summary['fixed_epoch_15']['std']} {aucs_ep15}")
    logger.info(f"  Fixed Epoch 10:           {summary['fixed_epoch_10']['mean']} +/- {summary['fixed_epoch_10']['std']} {aucs_ep10}")
    logger.info(f"  Validation Peeking:       {summary['val_peeking']['mean']} +/- {summary['val_peeking']['std']} {aucs_peek}")
    logger.info(f"  Validation Peeking Gap:   {summary['peeking_inflation']:+.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 5.75: Establish Noise Floor & Re-evaluate Stage 5.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Random seeds to evaluate.")
    parser.add_argument("--ref-seed", type=int, default=42, help="Reference seed for Stage 5 comparison.")
    parser.add_argument("--config", type=str, default="configs/hierarchical_mamba.yaml")
    parser.add_argument("--patient-config", type=str, default="configs/patient_mamba.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--cached-task1-json", type=str, default=None, help="Path to precomputed Task 1 seed summaries JSON.")
    args = parser.parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{timestamp}_noise_floor_study"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "training.log")
    logger.info("================================================================================")
    logger.info("STAGE 5.75: ESTABLISH THE NOISE FLOOR (Gate Before Stage 6)")
    logger.info("================================================================================")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Evaluating seeds: {args.seeds} (Reference seed: {args.ref_seed})")

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    with open(args.config, "r", encoding="utf-8") as f:
        hier_config = yaml.safe_load(f)
    hier_config["training"]["epochs"] = args.epochs

    with open(args.patient_config, "r", encoding="utf-8") as f:
        patient_config = yaml.safe_load(f)

    # -------------------------------------------------------------------------
    # TASK 1: Multi-Seed Variance Check across Seeds
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("TASK 1: MULTI-SEED VARIANCE CHECK")
    logger.info("=" * 80)

    ref_ckpt_dir = Path("runs/stage4_fixed_checkpoints")
    seed_summaries = {}

    if args.cached_task1_json and Path(args.cached_task1_json).exists():
        logger.info(f"Loading precomputed Task 1 seed summaries from {args.cached_task1_json}...")
        with open(args.cached_task1_json, "r", encoding="utf-8") as f:
            raw_s = json.load(f)
        seed_summaries = {int(k): v for k, v in raw_s.items()}
    else:
        for s in args.seeds:
            is_ref = (s == args.ref_seed)
            seed_res = run_seed_cv(
                seed=s,
                config=hier_config,
                device=device,
                logger=logger,
                is_reference_seed=is_ref,
                ref_ckpt_dir=ref_ckpt_dir if is_ref else None,
            )
            seed_summaries[s] = seed_res

    # Compute Noise Floor across seeds
    means_ep15 = [seed_summaries[s]["fixed_epoch_15"]["mean"] for s in args.seeds]
    means_peek = [seed_summaries[s]["val_peeking"]["mean"] for s in args.seeds]

    noise_floor_spread = float(np.max(means_ep15) - np.min(means_ep15))
    noise_floor_std = float(np.std(means_ep15))
    mean_across_seeds_ep15 = float(np.mean(means_ep15))

    logger.info("\n" + "=" * 80)
    logger.info("TASK 1 NOISE FLOOR SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Seed':<8} | {'Fixed Ep15 AUC (Honest)':<26} | {'Val-Peeking AUC (Optimistic)':<28} | {'Peeking Gap':<12}")
    logger.info("-" * 80)
    for s in args.seeds:
        e15 = seed_summaries[s]["fixed_epoch_15"]["mean"]
        pk = seed_summaries[s]["val_peeking"]["mean"]
        gap = seed_summaries[s]["peeking_inflation"]
        logger.info(f"{s:<8} | {e15:<26.4f} | {pk:<28.4f} | {gap:+.4f}")
    logger.info("-" * 80)
    logger.info(f"Multi-Seed Mean (Fixed Ep15):  {mean_across_seeds_ep15:.4f}")
    logger.info(f"Multi-Seed Std (Fixed Ep15):   {noise_floor_std:.4f}")
    logger.info(f"Multi-Seed Spread (Max - Min): {noise_floor_spread:.4f}")
    logger.info(f"*** EMPIRICAL NOISE FLOOR: {noise_floor_spread:.4f} (spread) / {noise_floor_std:.4f} (std) ***")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # TASK 2: Non-Peeking Baseline Establishment
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("TASK 2: ESTABLISH THE HONEST STAGE 4 BASELINE (NO PEEKING)")
    logger.info("=" * 80)

    ref_summary = seed_summaries[args.ref_seed]
    honest_baseline_ep15 = ref_summary["fixed_epoch_15"]
    logger.info(f"Canonical Reference Seed: {args.ref_seed}")
    logger.info(f"Honest 5-Fold Baseline (Fixed Epoch 15): {honest_baseline_ep15['mean']} +/- {honest_baseline_ep15['std']}")
    logger.info(f"Per-Fold AUCs: {honest_baseline_ep15['per_fold']}")
    logger.info(f"Validation-Peeking Baseline (Historical): {ref_summary['val_peeking']['mean']} +/- {ref_summary['val_peeking']['std']}")
    logger.info(f"Validation Peeking Inflation: {ref_summary['peeking_inflation']:+.4f}")

    # -------------------------------------------------------------------------
    # TASK 3: Extract Embeddings from Fixed Epoch Checkpoints & Re-evaluate Stage 5 ONCE
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("TASK 3: EXTRACT EMBEDDINGS & RE-RUN STAGE 5 COMPARISON ONCE")
    logger.info("=" * 80)

    fixed_embeddings_dir = Path("data/processed/embeddings_fixed_ep15")
    logger.info(f"Extracting embeddings using Seed {args.ref_seed} Fixed Epoch 15 checkpoints into {fixed_embeddings_dir}...")

    with open(hier_config["data"]["patient_index"], "r", encoding="utf-8") as f:
        patient_index = json.load(f)
    with open(hier_config["data"]["splits"], "r", encoding="utf-8") as f:
        folds_data = json.load(f)

    for f_idx in range(5):
        ckpt_path = ref_ckpt_dir / f"best_model_fold_{f_idx}.pt"
        assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
        extract_fold_embeddings(
            fold_idx=f_idx,
            checkpoint_path=ckpt_path,
            config=hier_config,
            patient_index=patient_index,
            folds_data=folds_data,
            output_dir=fixed_embeddings_dir,
            device=device,
            logger=logger,
        )

    # Configure Stage 5 to use the newly extracted fixed-epoch embeddings and checkpoints
    patient_config["data"]["embeddings_dir"] = str(fixed_embeddings_dir)
    patient_config["training"]["checkpoints_dir"] = str(ref_ckpt_dir)
    patient_config["training"]["warm_start_head"] = True

    # Compute Task 1 Control on the fixed checkpoints
    logger.info("\n--- Computing Stage 4 Control on Fixed Epoch 15 Checkpoints ---")
    task1_control = compute_stage4_control(patient_config)
    s4_vol_logit = task1_control["stage4_volume_logit_mean"]
    s4_emb_head = task1_control["stage4_embedding_mean_then_head"]
    logger.info(f"Stage 4 Volume Logit Mean (Control A): {s4_vol_logit['mean']} +/- {s4_vol_logit['std']} {s4_vol_logit['per_fold']}")
    logger.info(f"Stage 4 Embedding Mean then Head (Control B): {s4_emb_head['mean']} +/- {s4_emb_head['std']} {s4_emb_head['per_fold']}")

    # Re-evaluate all 4 aggregators
    stage5_results = {}
    for model_cfg in patient_config["models"]:
        res = run_model_cv(model_cfg, patient_config, device, logger)
        stage5_results[model_cfg["name"]] = res

    # Paired Per-Fold Comparison Table
    logger.info("\n" + "=" * 105)
    logger.info("STAGE 5.75 RECONCILED PAIRED COMPARISON (FIXED EPOCH 15 PROTOCOL)")
    logger.info("=" * 105)
    header = f"{'Fold':<6} | {'Stage 4 Vol Logit':<18} | {'Stage 4 Emb Head':<18} | {'Mean Pool':<12} | {'Attention':<12} | {'Patient Mamba':<14} | {'Mamba Delta':<12}"
    logger.info(header)
    logger.info("-" * len(header))

    s4_vol_aucs = s4_vol_logit["per_fold"]
    s4_emb_aucs = s4_emb_head["per_fold"]
    mamba_aucs = stage5_results["patient_mamba"]["classification_default_tau_0_50"]["roc_auc"]["per_fold"]
    mean_aucs = stage5_results["mean_pooling"]["classification_default_tau_0_50"]["roc_auc"]["per_fold"]
    att_aucs = stage5_results["attention_pooling"]["classification_default_tau_0_50"]["roc_auc"]["per_fold"]

    for f_idx in range(5):
        s4_v = s4_vol_aucs[f_idx]
        s4_e = s4_emb_aucs[f_idx]
        pm = mamba_aucs[f_idx]
        mp = mean_aucs[f_idx]
        ap = att_aucs[f_idx]
        delta = pm - s4_v
        logger.info(f"Fold {f_idx:<1} | {s4_v:<18.4f} | {s4_e:<18.4f} | {mp:<12.4f} | {ap:<12.4f} | {pm:<14.4f} | {delta:+.4f}")

    logger.info("-" * len(header))
    m_s4_v = s4_vol_logit["mean"]
    m_s4_e = s4_emb_head["mean"]
    m_pm = stage5_results["patient_mamba"]["classification_default_tau_0_50"]["roc_auc"]["mean"]
    m_mp = stage5_results["mean_pooling"]["classification_default_tau_0_50"]["roc_auc"]["mean"]
    m_ap = stage5_results["attention_pooling"]["classification_default_tau_0_50"]["roc_auc"]["mean"]
    delta_mamba_vs_s4 = m_pm - m_s4_v
    delta_mamba_vs_mean = m_pm - m_mp
    logger.info(f"{'Mean':<6} | {m_s4_v:<18.4f} | {m_s4_e:<18.4f} | {m_mp:<12.4f} | {m_ap:<12.4f} | {m_pm:<14.4f} | {delta_mamba_vs_s4:+.4f}")
    logger.info("=" * 105)

    # -------------------------------------------------------------------------
    # DECISION: Test Against Noise Floor
    # -------------------------------------------------------------------------
    clears_noise_floor_s4 = (delta_mamba_vs_s4 > noise_floor_spread)
    clears_noise_floor_mean = (delta_mamba_vs_mean > noise_floor_spread)

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 5.75 GATE DECISION: TESTING AGAINST THE NOISE FLOOR")
    logger.info("=" * 80)
    logger.info(f"Task 1 Empirical Noise Floor (Spread): {noise_floor_spread:.4f}")
    logger.info(f"Task 1 Empirical Noise Floor (Std):    {noise_floor_std:.4f}")
    logger.info(f"Patient Mamba vs Stage 4 Baseline Delta:  {delta_mamba_vs_s4:+.4f}")
    logger.info(f"Patient Mamba vs Mean Pooling Delta:     {delta_mamba_vs_mean:+.4f}")
    logger.info("-" * 80)

    if clears_noise_floor_s4 and clears_noise_floor_mean:
        verdict = "CLEARS_NOISE_FLOOR"
        logger.info("[VERDICT]: Patient Mamba's advantage CLEARS the multi-seed noise floor.")
        logger.info("Proceed to Stage 6 with this honestly-earned sequence modeling win as justification.")
    else:
        verdict = "WITHIN_NOISE_FLOOR"
        logger.info("[VERDICT]: Patient Mamba's advantage is NOT distinguishable from the multi-seed noise floor.")
        logger.info("Report as a legitimate negative result: at N=92 patients, sequence modeling architecture")
        logger.info("cannot overcome small-sample dataset variance. Stage 6 (Self-Supervised Pretraining) is")
        logger.info("justified targeting encoder generalization on unlabeled data, rather than probe capacity.")
    logger.info("=" * 80)

    # Save outputs
    output_payload = {
        "timestamp": timestamp,
        "stage": 5.75,
        "seeds": args.seeds,
        "reference_seed": args.ref_seed,
        "task1_multi_seed_noise_floor": {
            "noise_floor_spread": round(noise_floor_spread, 4),
            "noise_floor_std": round(noise_floor_std, 4),
            "mean_across_seeds_fixed_ep15": round(mean_across_seeds_ep15, 4),
            "per_seed_summaries": {
                str(s): {
                    "fixed_epoch_15": seed_summaries[s].get("fixed_epoch_15"),
                    "fixed_epoch_10": seed_summaries[s].get("fixed_epoch_10"),
                    "min_train_loss": seed_summaries[s].get("min_train_loss"),
                    "val_peeking": seed_summaries[s].get("val_peeking"),
                    "peeking_inflation": seed_summaries[s].get("peeking_inflation"),
                }
                for s in args.seeds
            },
        },
        "task2_honest_baseline": {
            "canonical_reference_seed": args.ref_seed,
            "honest_fixed_epoch_15": honest_baseline_ep15,
            "stage4_volume_logit_mean": s4_vol_logit,
            "stage4_embedding_mean_then_head": s4_emb_head,
        },
        "task3_reconciled_stage5": {
            "aggregators": stage5_results,
            "delta_mamba_vs_stage4_baseline": round(delta_mamba_vs_s4, 4),
            "delta_mamba_vs_mean_pooling": round(delta_mamba_vs_mean, 4),
        },
        "decision": {
            "verdict": verdict,
            "noise_floor_spread": round(noise_floor_spread, 4),
            "delta_mamba_vs_stage4": round(delta_mamba_vs_s4, 4),
            "delta_mamba_vs_mean_pool": round(delta_mamba_vs_mean, 4),
            "clears_noise_floor_stage4": clears_noise_floor_s4,
            "clears_noise_floor_mean_pool": clears_noise_floor_mean,
        },
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"hierarchical": hier_config, "patient_mamba": patient_config}, f)

    logger.info(f"\nAll Stage 5.75 artifacts written to {run_dir}")


if __name__ == "__main__":
    main()
