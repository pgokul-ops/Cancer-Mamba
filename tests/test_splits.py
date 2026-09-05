"""
tests/test_splits.py

Unit tests verifying patient-level 5-fold cross-validation split properties:
- No patient appears in more than one fold (strict zero-leakage).
- Every patient in manifest has an assigned fold.
- All studies/series belonging to a patient inherit that patient's fold.
- Class balance preservation across folds.
"""

import json
from pathlib import Path
import pytest


SPLITS_PATH = Path("data/splits/folds_v1.json")
MANIFEST_PATH = Path("data/manifests/manifest_v1.json")
SERIES_SELECTION_PATH = Path("data/manifests/series_selection.json")


def test_splits_file_exists():
    assert SPLITS_PATH.exists(), f"Splits file not found at {SPLITS_PATH}"


def test_no_patient_leakage_across_folds():
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    folds = splits_data["folds"]
    seen_patients = set()
    total_in_folds = 0

    for fold_name, patient_list in folds.items():
        fold_set = set(patient_list)
        # Check no duplicates within fold
        assert len(fold_set) == len(patient_list), f"Duplicate patients found within {fold_name}"

        # Check disjointness across folds
        overlap = seen_patients & fold_set
        assert len(overlap) == 0, f"Data leakage detected! Patients {overlap} appear in multiple folds!"

        seen_patients.update(fold_set)
        total_in_folds += len(patient_list)

    assert total_in_folds == 92, f"Expected 92 patients across folds, found {total_in_folds}"
    assert len(seen_patients) == 92, f"Expected 92 unique patients, found {len(seen_patients)}"


def test_all_manifest_patients_covered():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    manifest_patients = set(s["patient_id"] for s in manifest["series"])
    split_patients = set(splits_data["patient_to_fold"].keys())

    missing = manifest_patients - split_patients
    assert len(missing) == 0, f"Manifest patients missing from splits: {missing}"

    extra = split_patients - manifest_patients
    assert len(extra) == 0, f"Unexpected extra patients in splits: {extra}"


def test_series_inherit_patient_fold():
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    patient_to_fold = splits_data["patient_to_fold"]

    with open(SERIES_SELECTION_PATH, "r", encoding="utf-8") as f:
        selection_data = json.load(f)

    # Verify that every selected series belongs to exactly the patient's fold
    for pid, studies in selection_data["selection_by_patient"].items():
        assert pid in patient_to_fold, f"Patient {pid} missing from fold assignments"
        expected_fold = patient_to_fold[pid]

        for sid, series_list in studies.items():
            for s in series_list:
                # The series implicitly and explicitly belongs to the patient's fold
                assert s["patient_id"] == pid
                series_fold = patient_to_fold[s["patient_id"]]
                assert series_fold == expected_fold, (
                    f"Series {s['series_id']} of patient {pid} assigned to fold {series_fold}, "
                    f"expected patient fold {expected_fold}"
                )


def test_stratification_balance():
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    stats = splits_data["fold_statistics"]
    for fold_name, f_stats in stats.items():
        v2 = f_stats["label_v2_counts"]
        # Exactly 9 class 1 in each of the 5 folds (45 / 5 = 9)
        assert v2["class_1"] == 9, f"{fold_name} has {v2['class_1']} class 1, expected 9"
        # Either 5 or 6 class 0 in each fold (27 / 5 = 5.4)
        assert v2["class_0"] in (5, 6), f"{fold_name} has {v2['class_0']} class 0, expected 5 or 6"
        # Exactly 4 unlabeled survival patients in each fold (20 / 5 = 4)
        assert v2["unlabeled"] == 4, f"{fold_name} has {v2['unlabeled']} unlabeled, expected 4"
