#!/usr/bin/env python3
"""
datasets/patient_sequence_dataset.py

Patient-level sequence dataset for volumetric cancer imaging.
Constructs ordered sequences of volume embeddings for each patient:
  - Temporal ordering by study/acquisition date (MM-DD-YYYY extracted from manifest paths).
  - Tie-breaking by clinically-motivated contrast phase hierarchy:
      non_contrast -> arterial -> portal_venous -> primary_routine -> contrast_enhanced_general -> delayed -> lung_window -> chest_std.
  - Length-1 sequences for single-volume patients (N=33).
  - Batch collation with boolean padding masks for sequences up to max length (12).
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


PHASE_PRIORITY: Dict[str, int] = {
    "non_contrast": 0,
    "arterial": 1,
    "portal_venous": 2,
    "primary_routine": 3,
    "contrast_enhanced_general": 4,
    "delayed": 5,
    "lung_window": 6,
    "chest_std": 7,
}


def parse_manifest_dates(manifest_path: Union[str, Path]) -> Dict[str, datetime]:
    """Extracts acquisition/study dates from series paths in manifest_v1.json."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    date_pattern = re.compile(r"(\d{2}-\d{2}-\d{4})")
    series_dates = {}
    for s in manifest.get("series", []):
        sid = s.get("series_id")
        path_str = s.get("path", "")
        match = date_pattern.search(path_str)
        if match:
            try:
                dt = datetime.strptime(match.group(1), "%m-%d-%Y")
                series_dates[sid] = dt
            except ValueError:
                series_dates[sid] = datetime.min
        else:
            series_dates[sid] = datetime.min
    return series_dates


class PatientSequenceDataset(Dataset):
    """
    Dataset returning ordered sequences of volume embeddings per patient.
    """

    def __init__(
        self,
        patient_index_path: Union[str, Path] = "data/processed/patient_index.json",
        splits_path: Union[str, Path] = "data/splits/folds_v1.json",
        manifest_path: Union[str, Path] = "data/manifests/manifest_v1.json",
        embeddings_dir: Union[str, Path] = "data/processed/embeddings",
        fold_idx: int = 0,
        split_type: str = "train",  # "train", "val", or "all"
        preload: bool = True,
        labeled_only: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.patient_index_path = Path(patient_index_path)
        self.splits_path = Path(splits_path)
        self.manifest_path = Path(manifest_path)
        self.embeddings_dir = Path(embeddings_dir) / f"fold_{fold_idx}"
        self.fold_idx = fold_idx
        self.split_type = split_type
        self.preload = preload
        self.labeled_only = labeled_only
        self.logger = logger or logging.getLogger(__name__)

        with open(self.patient_index_path, "r", encoding="utf-8") as f:
            self.patient_index = json.load(f)
        with open(self.splits_path, "r", encoding="utf-8") as f:
            self.splits = json.load(f)

        self.series_dates = parse_manifest_dates(self.manifest_path)

        patient_to_fold = self.splits["patient_to_fold"]

        # Filter patients according to split_type
        self.patients = []
        for pid, pdata in self.patient_index.items():
            p_fold = patient_to_fold.get(pid, -1)
            if self.split_type == "val" and p_fold != self.fold_idx:
                continue
            elif self.split_type == "train" and p_fold == self.fold_idx:
                continue
            # For survival, label_v2 can be -1 (unlabeled 20 patients)
            if self.labeled_only and pdata.get("label_v2", -1) not in (0, 1):
                continue
            self.patients.append(pid)

        # Build ordered sequences for each patient
        self.patient_sequences: Dict[str, List[Dict[str, Any]]] = {}
        for pid in self.patients:
            vols = self.patient_index[pid]["volumes"]
            sorted_vols = sorted(
                vols,
                key=lambda v: (
                    self.series_dates.get(v["series_id"], datetime.min),
                    PHASE_PRIORITY.get(v.get("role_tag", ""), 99),
                    v["series_id"],
                ),
            )
            self.patient_sequences[pid] = sorted_vols

        # Optional preloading into memory
        self.preloaded_embeddings: Dict[str, np.ndarray] = {}
        if self.preload and self.embeddings_dir.exists():
            for pid in self.patients:
                for v in self.patient_sequences[pid]:
                    study_id = v["study_id"]
                    series_id = v["series_id"]
                    emb_path = self.embeddings_dir / pid / f"{study_id}_{series_id}.npy"
                    if emb_path.exists():
                        self.preloaded_embeddings[f"{pid}_{study_id}_{series_id}"] = np.load(emb_path)

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid = self.patients[idx]
        pdata = self.patient_index[pid]
        vols = self.patient_sequences[pid]

        seq_embs = []
        for v in vols:
            study_id = v["study_id"]
            series_id = v["series_id"]
            cache_key = f"{pid}_{study_id}_{series_id}"
            if cache_key in self.preloaded_embeddings:
                emb = self.preloaded_embeddings[cache_key]
            else:
                emb_path = self.embeddings_dir / pid / f"{study_id}_{series_id}.npy"
                if emb_path.exists():
                    emb = np.load(emb_path)
                else:
                    # Fallback zero vector for testing before extraction completes
                    emb = np.zeros(256, dtype=np.float32)
            seq_embs.append(emb)

        seq_tensor = torch.tensor(np.stack(seq_embs), dtype=torch.float32)  # (L, 256)
        label_v2 = float(pdata.get("label_v2", -1))
        duration = float(pdata.get("survival_duration_days", 0.0))
        event = float(pdata.get("survival_event", 0.0))

        return {
            "patient_id": pid,
            "embeddings": seq_tensor,
            "num_volumes": len(vols),
            "label_v2": label_v2,
            "survival_duration": duration,
            "survival_event": event,
        }


def collate_patient_sequences(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collates patient samples with variable volume lengths into padded tensors.
    Returns:
        embeddings: (B, max_len, 256)
        mask: (B, max_len) - True for valid tokens, False for padded
        lengths: (B,) - actual volume count per patient
        labels: (B,) - classification label (0, 1, or -1)
        durations: (B,) - continuous survival time in days
        events: (B,) - event indicator (1=death, 0=censored)
        patient_ids: List[str]
    """
    batch_size = len(batch)
    lengths = [sample["num_volumes"] for sample in batch]
    max_len = max(lengths)
    d_model = batch[0]["embeddings"].size(-1)

    padded_embs = torch.zeros(batch_size, max_len, d_model, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.float32)
    durations = torch.zeros(batch_size, dtype=torch.float32)
    events = torch.zeros(batch_size, dtype=torch.float32)
    patient_ids = []

    for i, sample in enumerate(batch):
        seq = sample["embeddings"]
        length = sample["num_volumes"]
        padded_embs[i, :length] = seq
        mask[i, :length] = True
        labels[i] = sample["label_v2"]
        durations[i] = sample["survival_duration"]
        events[i] = sample["survival_event"]
        patient_ids.append(sample["patient_id"])

    return {
        "embeddings": padded_embs,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": labels,
        "durations": durations,
        "events": events,
        "patient_ids": patient_ids,
    }
