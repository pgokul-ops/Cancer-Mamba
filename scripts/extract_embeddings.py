#!/usr/bin/env python3
"""
scripts/extract_embeddings.py

Extracts 256-dim pre-classifier volume embeddings from Stage 4 Hierarchical 3D Mamba.
Executes per-fold extraction:
  For each fold f in 0..4:
    - Checks or trains the Stage 4 Hierarchical encoder on fold f's training set (saving checkpoint).
    - Extracts pre-classifier [mean, max] pooled embeddings (256-dim) for all volumes.
    - Explicitly asserts that fold f's validation patients are NEVER in fold f's training set.
    - Saves embeddings to data/processed/embeddings/fold_{f}/{patient_id}/{study_id}_{series_id}.npy.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from datasets.volume_dataset import VolumeDataset
from models.hierarchical_mamba import HierarchicalMamba3D
from training.trainer import Trainer


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ExtractEmbeddings")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def build_model(config: Dict[str, Any]) -> HierarchicalMamba3D:
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


def train_fold_checkpoint(
    fold_idx: int,
    config: Dict[str, Any],
    device: torch.device,
    save_path: Path,
    logger: logging.Logger,
) -> None:
    """Trains a fold model and saves the best checkpoint."""
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

    model = build_model(config).to(device)
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    pos_weight = config["training"].get("pos_weight")
    use_amp = config["training"].get("amp", True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pos_weight_tensor = torch.tensor([pos_weight], device=device) if pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc = -1.0
    best_state_dict = None
    best_epoch = 0

    logger.info(f"Training Fold {fold_idx} ({epochs} epochs) to obtain best checkpoint...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, _, _ in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y)

        # Validation
        model.eval()
        predictions = []
        with torch.no_grad():
            for x, y, pids, ssids in val_loader:
                x = x.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x).squeeze(1).cpu()
                for logit, label, pid, ssid in zip(logits, y, pids, ssids):
                    predictions.append({
                        "patient_id": pid,
                        "study_series_id": ssid,
                        "logit": float(logit),
                        "label": int(label),
                    })

        from evaluation.metrics import evaluate_predictions
        eval_res = evaluate_predictions(predictions)
        auc = eval_res["patient_level"].get("roc_auc")
        score = auc if auc is not None else eval_res["patient_level"]["balanced_accuracy"]

        if score > best_auc:
            best_auc = score
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"  Epoch {epoch:2d}/{epochs} | Val Patient AUC: {auc if auc else 'N/A'} (best: {best_auc:.4f} at ep {best_epoch})")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": best_epoch,
            "best_score": best_auc,
            "state_dict": best_state_dict,
            "fold": fold_idx,
        },
        save_path,
    )
    logger.info(f"Saved Fold {fold_idx} checkpoint to {save_path} (Best Score: {best_auc:.4f} at epoch {best_epoch})")


def extract_fold_embeddings(
    fold_idx: int,
    checkpoint_path: Path,
    config: Dict[str, Any],
    patient_index: Dict[str, Any],
    folds_data: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """Extracts pre-classifier embeddings for all volumes using the specified fold model."""
    logger.info(f"Loading checkpoint {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    patient_to_fold = folds_data["patient_to_fold"]
    val_patients = set([p for p, f in patient_to_fold.items() if f == fold_idx])
    train_patients = set([p for p, f in patient_to_fold.items() if f != fold_idx])

    # STRICT LEAKAGE ASSERTION
    overlap = val_patients.intersection(train_patients)
    assert len(overlap) == 0, f"Critical Data Leakage in Fold {fold_idx}! Overlapping patients: {overlap}"

    logger.info(f"Fold {fold_idx}: {len(val_patients)} validation patients, {len(train_patients)} training patients.")
    logger.info(f"Strict leakage check passed: 0 patient overlap between train and val.")

    fold_out_dir = output_dir / f"fold_{fold_idx}"
    fold_out_dir.mkdir(parents=True, exist_ok=True)

    extracted_records = []
    start_time = time.time()
    total_volumes = 0

    with torch.no_grad():
        for pid, pdata in patient_index.items():
            p_fold = patient_to_fold.get(pid, -1)
            is_val = (p_fold == fold_idx)

            p_dir = fold_out_dir / pid
            p_dir.mkdir(parents=True, exist_ok=True)

            for vol in pdata["volumes"]:
                study_id = vol["study_id"]
                series_id = vol["series_id"]
                cached_path = Path(vol["cached_path"])

                if not cached_path.exists():
                    raise FileNotFoundError(f"Missing cached volume: {cached_path}")

                payload = torch.load(cached_path, map_location="cpu")
                vol_tensor = payload["volume"].unsqueeze(0).to(device)  # (1, 1, D, H, W)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    feat = model.forward_features(vol_tensor)  # (1, 256)

                feat_np = feat.cpu().squeeze(0).float().numpy()

                out_filename = f"{study_id}_{series_id}.npy"
                out_path = p_dir / out_filename
                np.save(out_path, feat_np)

                extracted_records.append({
                    "patient_id": pid,
                    "study_id": study_id,
                    "series_id": series_id,
                    "role_tag": vol.get("role_tag"),
                    "is_val": is_val,
                    "patient_fold": p_fold,
                    "embedding_path": str(out_path.relative_to(output_dir)),
                    "feature_dim": feat_np.shape[0],
                })
                total_volumes += 1

    duration = time.time() - start_time
    logger.info(
        f"Fold {fold_idx}: Successfully extracted {total_volumes} volume embeddings in {duration:.1f}s ({total_volumes/duration:.1f} vol/s)."
    )

    manifest_path = fold_out_dir / "embedding_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fold": fold_idx,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "checkpoint_epoch": ckpt["epoch"],
                "checkpoint_best_score": ckpt.get("best_score", 0.0),
                "total_volumes": total_volumes,
                "total_patients": len(patient_index),
                "num_val_patients": len(val_patients),
                "num_train_patients": len(train_patients),
                "volumes": extracted_records,
            },
            f,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(description="Extract Stage 4 volume embeddings per fold.")
    parser.add_argument("--config", type=str, default="configs/hierarchical_mamba.yaml")
    parser.add_argument("--output-dir", type=str, default="data/processed/embeddings")
    parser.add_argument("--checkpoints-dir", type=str, default="runs/stage4_checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-retrain", action="store_true", help="Force retrain checkpoints even if existing.")
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("=== Stage 5: Volume Embedding Extraction (Leakage-Free Per Fold) ===")

    # Reproducibility seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    with open(config["data"]["patient_index"], "r") as f:
        patient_index = json.load(f)
    with open(config["data"]["splits"], "r") as f:
        folds_data = json.load(f)

    output_dir = Path(args.output_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    all_folds = config["cv"]["folds"]

    total_start = time.time()

    for fold_idx in all_folds:
        ckpt_path = checkpoints_dir / f"best_model_fold_{fold_idx}.pt"
        if not ckpt_path.exists() or args.force_retrain:
            train_fold_checkpoint(
                fold_idx=fold_idx,
                config=config,
                device=device,
                save_path=ckpt_path,
                logger=logger,
            )
        else:
            logger.info(f"Using existing checkpoint: {ckpt_path}")

        extract_fold_embeddings(
            fold_idx=fold_idx,
            checkpoint_path=ckpt_path,
            config=config,
            patient_index=patient_index,
            folds_data=folds_data,
            output_dir=output_dir,
            device=device,
            logger=logger,
        )

    total_time = time.time() - total_start
    logger.info(f"\nAll 5 folds completed in {total_time:.1f} seconds (~{total_time/60:.1f} minutes).")
    logger.info(f"Embeddings saved under: {output_dir}")


if __name__ == "__main__":
    main()
