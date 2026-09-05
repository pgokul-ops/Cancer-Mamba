"""
datasets/volume_dataset.py

PyTorch Dataset reading cached volumetric CT tensors for Cancer-Mamba Stage 2.
Filters strictly to patients where label_v2 is available (45:27 cohort).
Yields (volume_tensor, label_v2, patient_id, study_series_id) per cached volume.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


class VolumeDataset(Dataset):
    """
    Dataset that loads individual standardized 3D volume tensors from data/processed/.

    Args:
        patient_index_path: Path to data/processed/patient_index.json.
        splits_path: Path to data/splits/folds_v1.json.
        folds: List of fold indices to include (e.g. [1, 2, 3, 4] for train, [0] for val).
               If None, includes all folds.
        transform: Optional transform callable applied to volume tensor.
        verbose: If True, prints fold sample and patient counts and volume histogram.
    """

    def __init__(
        self,
        patient_index_path: Union[str, Path] = "data/processed/patient_index.json",
        splits_path: Union[str, Path] = "data/splits/folds_v1.json",
        folds: Optional[List[int]] = None,
        transform=None,
        verbose: bool = True,
    ):
        super().__init__()
        self.patient_index_path = Path(patient_index_path)
        self.splits_path = Path(splits_path)
        self.folds = set(folds) if folds is not None else None
        self.transform = transform

        if not self.patient_index_path.exists():
            raise FileNotFoundError(f"Patient index not found: {self.patient_index_path}")
        if not self.splits_path.exists():
            raise FileNotFoundError(f"Splits file not found: {self.splits_path}")

        with open(self.patient_index_path, "r", encoding="utf-8") as f:
            self.patient_index: Dict[str, Any] = json.load(f)

        with open(self.splits_path, "r", encoding="utf-8") as f:
            self.splits_data: Dict[str, Any] = json.load(f)

        self.patient_to_fold: Dict[str, int] = self.splits_data["patient_to_fold"]

        # Filter to patients where label_v2 is not None and within requested folds
        self.items: List[Dict[str, Any]] = []
        self.patients_in_dataset: set = set()
        patient_vol_counts: Counter = Counter()

        for pid, pdata in self.patient_index.items():
            label_v2 = pdata.get("label_v2")
            if label_v2 is None:
                continue

            fold = self.patient_to_fold.get(pid)
            if self.folds is not None and fold not in self.folds:
                continue

            # Self-check assertion: verify patient fold matches folds_v1.json
            assert fold == pdata.get("fold"), (
                f"Fold mismatch for patient {pid}: splits has {fold}, patient_index has {pdata.get('fold')}"
            )

            volumes = pdata.get("volumes", [])
            for vol_meta in volumes:
                cached_path = Path(vol_meta["cached_path"])
                if not cached_path.is_absolute() and not cached_path.exists():
                    cached_path = Path.cwd() / cached_path

                study_series_id = f"{vol_meta['study_id']}_{vol_meta['series_id']}"
                self.items.append({
                    "file_path": cached_path,
                    "patient_id": pid,
                    "study_series_id": study_series_id,
                    "label_v2": int(label_v2),
                    "fold": fold,
                    "role_tag": vol_meta.get("role_tag", "unknown"),
                })
                self.patients_in_dataset.add(pid)
                patient_vol_counts[pid] += 1

        self.patient_vol_counts = patient_vol_counts

        if verbose:
            self._print_dataset_summary()

    def _print_dataset_summary(self):
        fold_str = f"folds {sorted(list(self.folds))}" if self.folds is not None else "all folds"
        print(f"\n[VolumeDataset Init: {fold_str}]")
        print(f"  Total Samples (Volumes): {len(self.items)}")
        print(f"  Total Unique Patients:   {len(self.patients_in_dataset)}")

        # Breakdown per fold included
        per_fold_samples = Counter(item["fold"] for item in self.items)
        per_fold_patients = Counter(self.patient_to_fold[pid] for pid in self.patients_in_dataset)
        for f in sorted(per_fold_samples.keys()):
            print(f"    Fold {f}: {per_fold_samples[f]} volumes from {per_fold_patients[f]} patients")

        # Class balance
        class_counts = Counter(item["label_v2"] for item in self.items)
        patient_class_counts = Counter(
            self.patient_index[pid]["label_v2"] for pid in self.patients_in_dataset
        )
        print(f"  Class Balance (Volumes):  Class 1: {class_counts.get(1, 0)}, Class 0: {class_counts.get(0, 0)}")
        print(f"  Class Balance (Patients): Class 1: {patient_class_counts.get(1, 0)}, Class 0: {patient_class_counts.get(0, 0)}")

        # Volume-per-patient histogram
        vol_hist = Counter(self.patient_vol_counts.values())
        print("  Volume-Count per Patient Histogram (Over-representation check):")
        for v_count, n_pats in sorted(vol_hist.items()):
            print(f"    {v_count} volume(s)/patient: {n_pats} patient(s) ({n_pats * v_count} total gradient contributions)")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        item = self.items[idx]
        file_path = item["file_path"]

        # Load PyTorch payload
        payload = torch.load(file_path, map_location="cpu")
        if isinstance(payload, dict) and "volume" in payload:
            volume_tensor = payload["volume"]
        elif isinstance(payload, torch.Tensor):
            volume_tensor = payload
        else:
            raise TypeError(f"Unexpected tensor payload format in {file_path}")

        # Ensure float32 and shape (1, D, H, W)
        if volume_tensor.ndim == 3:
            volume_tensor = volume_tensor.unsqueeze(0)
        volume_tensor = volume_tensor.float()

        if self.transform is not None:
            volume_tensor = self.transform(volume_tensor)

        label_tensor = torch.tensor(item["label_v2"], dtype=torch.float32)
        patient_id = item["patient_id"]
        study_series_id = item["study_series_id"]

        return volume_tensor, label_tensor, patient_id, study_series_id
