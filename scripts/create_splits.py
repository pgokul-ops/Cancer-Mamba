#!/usr/bin/env python3
"""
scripts/create_splits.py

Generates 5-fold cross-validation splits at the patient_id level for the 92 TCGA-OV patients.
- Stratifies the 72 patients with label_v2 (45:27 class balance).
- Evenly distributes the 20 unlabeled survival patients across the 5 folds.
- Ensures zero data leakage: each patient is assigned to exactly one validation fold.
- All studies and series of a patient inherit that patient's fold assignment.

Outputs:
  data/splits/folds_v1.json
"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def compute_patient_labels_and_survival(
    clinical_path: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    """
    Computes label_v1, label_v2, and survival parameters for all patients in manifest.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_patients = sorted(list(set(s["patient_id"] for s in manifest["series"])))

    clin_df = pd.read_csv(clinical_path)
    clin_df["patient_id"] = clin_df["Unnamed: 0"].astype(str).str.strip()

    # Filter to patients on disk
    df = clin_df[clin_df["patient_id"].isin(manifest_patients)].copy()

    # Calculate labels
    # label_v1 (original 63-patient cohort):
    #   DeadatFollowUp == True and OS < 1825 -> 1
    #   DeadatFollowUp == False and OS >= 1825 -> 0
    #   Others -> None
    def calc_label_v1(row):
        is_dead = row.get("DeadatFollowUp")
        os_days = row.get("OS")
        if pd.notna(is_dead) and is_dead and pd.notna(os_days) and os_days < 1825:
            return 1
        elif pd.notna(is_dead) and not is_dead and pd.notna(os_days) and os_days >= 1825:
            return 0
        return None

    # label_v2 (corrected 72-patient cohort):
    #   DeadatFollowUp == True and OS < 1825 -> 1
    #   OS >= 1825 -> 0 (all 5-year survivors, including those who died after 5 years)
    #   Others (censored < 1825) -> None
    def calc_label_v2(row):
        is_dead = row.get("DeadatFollowUp")
        os_days = row.get("OS")
        if pd.notna(is_dead) and is_dead and pd.notna(os_days) and os_days < 1825:
            return 1
        elif pd.notna(os_days) and os_days >= 1825:
            return 0
        return None

    df["label_v1"] = df.apply(calc_label_v1, axis=1)
    df["label_v2"] = df.apply(calc_label_v2, axis=1)
    df["survival_duration_days"] = pd.to_numeric(df["OS"], errors="coerce")
    df["survival_event"] = df["DeadatFollowUp"].apply(lambda x: 1 if bool(x) else 0)

    # Sort deterministically
    df = df.sort_values("patient_id").reset_index(drop=True)
    return df


def create_patient_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Creates stratified 5-fold CV splits on label_v2, distributing unlabeled patients evenly.
    """
    patient_fold_map: Dict[str, int] = {}

    # Separate labeled (label_v2 not None) vs unlabeled
    labeled_df = df[df["label_v2"].notna()].copy().reset_index(drop=True)
    unlabeled_df = df[df["label_v2"].isna()].copy().reset_index(drop=True)

    # Stratified split on labeled cohort (45:27)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(skf.split(labeled_df["patient_id"], labeled_df["label_v2"])):
        for idx in val_idx:
            pid = labeled_df.iloc[idx]["patient_id"]
            patient_fold_map[pid] = fold_idx

    # KFold split on unlabeled cohort (20 patients -> 4 per fold)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (_, val_idx) in enumerate(kf.split(unlabeled_df["patient_id"])):
        for idx in val_idx:
            pid = unlabeled_df.iloc[idx]["patient_id"]
            patient_fold_map[pid] = fold_idx

    # Build fold summary
    folds_detail = {f"fold_{i}": [] for i in range(n_splits)}
    fold_stats = {}

    for i in range(n_splits):
        pids = [pid for pid, f in patient_fold_map.items() if f == i]
        folds_detail[f"fold_{i}"] = sorted(pids)

        sub_df = df[df["patient_id"].isin(pids)]
        fold_stats[f"fold_{i}"] = {
            "total_patients": len(pids),
            "label_v2_counts": {
                "class_1": int((sub_df["label_v2"] == 1).sum()),
                "class_0": int((sub_df["label_v2"] == 0).sum()),
                "unlabeled": int(sub_df["label_v2"].isna().sum()),
            },
            "label_v1_counts": {
                "class_1": int((sub_df["label_v1"] == 1).sum()),
                "class_0": int((sub_df["label_v1"] == 0).sum()),
                "unlabeled": int(sub_df["label_v1"].isna().sum()),
            },
            "survival_events": int((sub_df["survival_event"] == 1).sum()),
            "survival_censored": int((sub_df["survival_event"] == 0).sum()),
        }

    return {
        "metadata": {
            "version": "v1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "n_splits": n_splits,
            "seed": seed,
            "total_patients": len(df),
            "stratification_target": "label_v2 (45:27 cohort)",
            "unlabeled_survival_patients": len(unlabeled_df),
            "no_held_out_test_set_rationale": (
                "With a small cohort of 92 patients, reserving a static 15-20% test set "
                "would drastically reduce training sample size to ~60-70 patients. Instead, "
                "nested 5-fold cross-validation is used, reporting cross-validated out-of-fold metrics."
            ),
        },
        "patient_to_fold": patient_fold_map,
        "folds": folds_detail,
        "fold_statistics": fold_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Create 5-fold patient-level splits")
    parser.add_argument(
        "--clinical",
        type=Path,
        default=Path("DATA/ClinicalData101515.csv"),
        help="Path to clinical CSV",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/manifest_v1.json"),
        help="Path to manifest JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/folds_v1.json"),
        help="Output folds JSON path",
    )
    parser.add_argument("--splits", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    df = compute_patient_labels_and_survival(args.clinical, args.manifest)
    splits_data = create_patient_folds(df, n_splits=args.splits, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)

    print(f"\nSuccessfully generated 5-fold split in: {args.output}")
    print(f"Total patients: {splits_data['metadata']['total_patients']}")
    for fold_name, stats in splits_data["fold_statistics"].items():
        v2 = stats["label_v2_counts"]
        print(f"  {fold_name}: {stats['total_patients']} patients | label_v2: [class_1={v2['class_1']}, class_0={v2['class_0']}, unl={v2['unlabeled']}] | events: {stats['survival_events']}")


if __name__ == "__main__":
    main()
