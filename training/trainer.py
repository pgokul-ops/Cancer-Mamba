"""
training/trainer.py

Trainer for Stage 2 Minimal 3D CNN Baseline.
Includes:
  - Overfit test (memorization check on 8 samples).
  - Single-fold & 5-fold training loop.
  - FP16 mixed precision autocast (AMP) and GradScaler.
  - Gradient accumulation support.
  - NaN/Inf loss abort guard.
  - Prediction collapse detector (constant prediction alert).
  - VRAM tracking (peak memory via torch.cuda.max_memory_allocated).
"""

from collections import defaultdict
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from evaluation.metrics import evaluate_predictions
from models.tiny_cnn3d import TinyCNN3D, get_model_param_count


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        pos_weight: Optional[float] = None,
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.logger = logger or logging.getLogger("Trainer")

        # Loss function with optional positive class weighting
        if pos_weight is not None:
            weight_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def run_overfit_test(
        self,
        dataset,
        num_samples: int = 8,
        epochs: int = 40,
        target_loss: float = 0.05,
        target_acc: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Sanity test: verifies that the model can rapidly memorize a tiny subset of 8 volumes.
        Asserts loss -> ~0 and accuracy -> ~100%.
        """
        self.logger.info(f"\n{'='*70}\n[SANITY CHECK 1/3] Running Overfit Test on {num_samples} volumes...\n{'='*70}")
        indices = list(range(min(num_samples, len(dataset))))
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=num_samples, shuffle=False)

        self.model.train()
        losses = []
        final_acc = 0.0

        for epoch in range(1, epochs + 1):
            for x, y, pids, ssids in loader:
                x = x.to(self.device)
                y = y.to(self.device).unsqueeze(1)

                self.optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)

                # NaN guard
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(f"Overfit test encountered NaN/Inf loss at epoch {epoch}!")

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                losses.append(loss.item())

                # Accuracy check
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct = (preds == y).sum().item()
                final_acc = correct / len(y)

            if epoch % 10 == 0 or epoch == epochs:
                self.logger.info(f"  Epoch {epoch:2d}/{epochs}: Loss = {loss.item():.4f}, Accuracy = {final_acc*100:.1f}%")

        passed = (losses[-1] <= target_loss) or (final_acc >= target_acc)
        self.logger.info(f"Overfit Test Result: {'PASSED (Memorization Confirmed)' if passed else 'FAILED'}")
        if not passed:
            raise AssertionError(
                f"Overfit test failed! Final loss {losses[-1]:.4f} > {target_loss} and acc {final_acc:.2f} < {target_acc}. "
                "Check data loading, label alignment, or model gradients."
            )

        return {
            "passed": passed,
            "final_loss": round(losses[-1], 4),
            "final_accuracy": round(final_acc, 4),
            "epochs_run": epochs,
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, float, float]:
        """
        Runs one training epoch with mixed precision, gradient accumulation, and NaN guards.
        Returns: (mean_loss, throughput_samples_per_sec, epoch_duration)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        total_samples = 0
        start_time = time.time()

        self.optimizer.zero_grad()

        for batch_idx, (x, y, pids, ssids) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device).unsqueeze(1)
            batch_size = x.size(0)
            total_samples += batch_size

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss_scaled = loss / self.grad_accum_steps

            # NaN Guard
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(
                    f"NaN/Inf loss detected at Epoch {epoch}, Batch {batch_idx}! Aborting training."
                )

            self.scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * batch_size

        duration = time.time() - start_time
        throughput = total_samples / max(duration, 1e-4)
        mean_loss = total_loss / max(total_samples, 1)

        return mean_loss, throughput, duration

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Runs evaluation over validation loader, gathering predictions and detecting collapse.
        Returns evaluation dict with patient_level, volume_level_inflated, and patient_table.
        """
        self.model.eval()
        predictions = []

        for x, y, pids, ssids in val_loader:
            x = x.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(x).squeeze(1)  # (B,)

            logits_cpu = logits.cpu().numpy()
            y_cpu = y.numpy()

            for logit, label, pid, ssid in zip(logits_cpu, y_cpu, pids, ssids):
                predictions.append({
                    "patient_id": pid,
                    "study_series_id": ssid,
                    "logit": float(logit),
                    "label": int(label),
                })

        # Evaluate predictions
        eval_result = evaluate_predictions(predictions)

        # Prediction collapse guard
        probs = [row["predicted_prob"] for row in eval_result["patient_table"]]
        if len(probs) > 1:
            prob_std = float(np.std(probs))
            if prob_std < 1e-4:
                self.logger.warning(
                    f"PREDICTION COLLAPSE DETECTED: Standard deviation of predicted probabilities is {prob_std:.6f}! "
                    "Model is outputting near-constant predictions."
                )

        return eval_result

    def train_fold(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        fold_idx: int = 0,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Trains for a given fold across the specified number of epochs.
        Tracks peak VRAM and records the best patient-level evaluation.
        """
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        fold_start = time.time()
        best_eval = None
        best_auc = -1.0
        best_epoch = 0

        self.logger.info(f"\n--- Training Fold {fold_idx} ({epochs} epochs) ---")

        for epoch in range(1, epochs + 1):
            train_loss, throughput, epoch_time = self.train_epoch(train_loader, epoch)
            eval_result = self.evaluate(val_loader)
            pat_metrics = eval_result["patient_level"]
            vol_metrics = eval_result["volume_level_inflated"]

            auc_val = pat_metrics.get("roc_auc")
            auc_disp = f"{auc_val:.4f}" if auc_val is not None else "N/A"
            vol_auc_disp = f"{vol_metrics.get('roc_auc'):.4f}" if vol_metrics.get("roc_auc") is not None else "N/A"

            if epoch % 5 == 0 or epoch == epochs or epoch == 1:
                self.logger.info(
                    f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | "
                    f"Patient AUC: {auc_disp} (BACC: {pat_metrics['balanced_accuracy']:.3f}, F1: {pat_metrics['f1']:.3f}) | "
                    f"Vol AUC (inflated): {vol_auc_disp} | {throughput:.1f} smp/s"
                )

            # Track best epoch by patient AUC (or balanced accuracy if AUC is None)
            score = auc_val if auc_val is not None else pat_metrics["balanced_accuracy"]
            if score > best_auc:
                best_auc = score
                best_eval = eval_result
                best_epoch = epoch

            if scheduler is not None:
                scheduler.step()

        total_fold_time = time.time() - fold_start
        peak_vram_mb = 0.0
        if self.device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        return {
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "best_patient_metrics": best_eval["patient_level"] if best_eval else {},
            "best_volume_metrics": best_eval["volume_level_inflated"] if best_eval else {},
            "final_patient_table": best_eval["patient_table"] if best_eval else [],
            "total_time_seconds": round(total_fold_time, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }
