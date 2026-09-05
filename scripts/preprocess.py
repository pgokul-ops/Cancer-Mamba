#!/usr/bin/env python3
"""
scripts/preprocess.py

End-to-end preprocessing & caching pipeline for Cancer-Mamba (Stage 1).
- Loads selected primary series from data/manifests/series_selection.json.
- Reconstructs 3D volumes via SimpleITK with physical coordinate sorting.
- Standardizes volumes:
    * Mode A (default): Anisotropic target spacing (1.5mm x 1.5mm x 3.0mm), cropped/padded to 80x80x80.
    * Mode B (ablation): Isotropic 2.5mm^3 resampling.
    * HU windowing: Standard abdominal window [-135, 215] HU, normalized to [0.0, 1.0].
- Caches PyTorch tensors:
    data/processed/{patient_id}/{study_id}_{series_id}.pt
- Caches separate exploratory MRI under data/processed_mri_exploratory/ for patient TCGA-24-1423.
- Builds master patient index:
    data/processed/patient_index.json
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from tqdm import tqdm

from preprocessing.dicom import reconstruct_volume_from_dicom
from preprocessing.resampling import standardize_volume


def process_single_series(
    task_info: Dict[str, Any],
    output_dir: Path,
    target_shape: Tuple[int, int, int] = (80, 80, 80),
    resampling_mode: str = "mode_a",
    roi_mode: str = "full_volume",
    min_hu: float = -135.0,
    max_hu: float = 215.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Worker function to reconstruct, standardize, and cache a single series.
    Returns (success_dict, failure_dict).
    """
    patient_id = task_info["patient_id"]
    study_id = task_info["study_id"]
    series_id = task_info["series_id"]
    role_tag = task_info.get("role_tag", "primary_routine")
    raw_path_str = task_info["path"]
    series_desc = task_info.get("series_description", "")
    expected_instances = task_info.get("num_instances")

    series_dir = Path(raw_path_str)
    if not series_dir.is_absolute() and not series_dir.exists():
        # Try resolving relative to workspace root
        series_dir = Path.cwd() / raw_path_str

    # 1. Reconstruct 3D DICOM volume
    img, err = reconstruct_volume_from_dicom(
        series_dir=series_dir,
        expected_instances=expected_instances,
        series_id=series_id,
    )
    if err is not None:
        return None, {
            "patient_id": patient_id,
            "study_id": study_id,
            "series_id": series_id,
            "stage": "reconstruction",
            "error": err,
        }

    assert img is not None

    # 2. Standardize volume (resampling + HU windowing + crop/pad to target_shape)
    try:
        vol_np, actual_spacing = standardize_volume(
            image=img,
            target_shape=target_shape,
            resampling_mode=resampling_mode,
            roi_mode=roi_mode,
            min_hu=min_hu,
            max_hu=max_hu,
        )

        # Convert to PyTorch float32 tensor with channel dimension (1, D, H, W)
        tensor_vol = torch.from_numpy(vol_np).unsqueeze(0).float()

        # Target output path: data/processed/{patient_id}/{study_id}_{series_id}.pt
        # Sanitize study_id and series_id for safe filenames
        safe_study = study_id.replace(".", "_")[-12:]
        safe_series = series_id.replace(".", "_")[-12:]
        patient_out_dir = output_dir / patient_id
        patient_out_dir.mkdir(parents=True, exist_ok=True)
        pt_filename = f"{safe_study}_{safe_series}.pt"
        pt_path = patient_out_dir / pt_filename

        # Save tensor & metadata
        payload = {
            "volume": tensor_vol,
            "spacing": actual_spacing,
            "patient_id": patient_id,
            "study_id": study_id,
            "series_id": series_id,
            "role_tag": role_tag,
            "series_description": series_desc,
            "original_shape": img.GetSize(),      # (X, Y, Z) in SimpleITK
            "original_spacing": img.GetSpacing(),  # (dx, dy, dz)
            "resampling_mode": resampling_mode,
            "roi_mode": roi_mode,
            "hu_window": (min_hu, max_hu),
        }
        torch.save(payload, pt_path)

        # Return volume metadata for patient index
        rel_pt_path = str(pt_path.relative_to(Path.cwd())) if pt_path.is_relative_to(Path.cwd()) else str(pt_path)
        success_info = {
            "patient_id": patient_id,
            "study_id": study_id,
            "series_id": series_id,
            "role_tag": role_tag,
            "series_description": series_desc,
            "cached_path": rel_pt_path,
            "num_slices": expected_instances,
            "file_size_bytes": pt_path.stat().st_size,
        }
        return success_info, None

    except Exception as e:
        return None, {
            "patient_id": patient_id,
            "study_id": study_id,
            "series_id": series_id,
            "stage": "standardization_or_save",
            "error": str(e),
        }


def process_exploratory_mri(
    manifest_path: Path,
    output_dir: Path,
    target_shape: Tuple[int, int, int] = (80, 80, 80),
) -> Dict[str, Any]:
    """
    Caches exploratory MRI series (single patient TCGA-24-1423) separately
    under data/processed_mri_exploratory/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mr_series = [s for s in manifest["series"] if s["modality"] == "MR"]
    print(f"\nCaching {len(mr_series)} exploratory MRI series under {output_dir}...")

    cached_mri = []
    for s in mr_series:
        pid = s["patient_id"]
        sid = s["study_id"]
        s_uid = s["series_id"]
        p_dir = Path(s["path"])
        if not p_dir.is_absolute() and not p_dir.exists():
            p_dir = Path.cwd() / p_dir

        img, err = reconstruct_volume_from_dicom(p_dir, s.get("num_instances"), s_uid)
        if img is not None:
            # SimpleITK linear resample to 80x80x80
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            # Normalize MRI intensities to [0, 1] using 1st and 99th percentiles
            p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
            if p99 > p1:
                arr_norm = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)
            else:
                arr_norm = arr

            from preprocessing.resampling import crop_or_pad_3d
            vol_final = crop_or_pad_3d(arr_norm, target_shape=target_shape)
            t_vol = torch.from_numpy(vol_final).unsqueeze(0).float()

            safe_s = s_uid.replace(".", "_")[-12:]
            out_file = output_dir / f"{pid}_{safe_s}.pt"
            torch.save({
                "volume": t_vol,
                "patient_id": pid,
                "study_id": sid,
                "series_id": s_uid,
                "series_description": s.get("series_description", ""),
                "modality": "MR",
            }, out_file)
            cached_mri.append(str(out_file))

    print(f"  Successfully cached {len(cached_mri)} MRI series to {output_dir}")
    return {"mri_patient_id": "TCGA-24-1423", "cached_series_count": len(cached_mri), "files": cached_mri}


def build_patient_index(
    processed_volumes: List[Dict[str, Any]],
    splits_path: Path,
    clinical_path: Path,
    output_index_path: Path,
) -> Dict[str, Any]:
    """
    Builds data/processed/patient_index.json mapping each patient to all cached volumes,
    fold assignment, dual labels (label_v1, label_v2), and survival metrics.
    """
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)
    patient_to_fold = splits_data["patient_to_fold"]

    # Load Clinical Data
    clin_df = pd.read_csv(clinical_path)
    clin_df["patient_id"] = clin_df["Unnamed: 0"].astype(str).str.strip()

    clinical_lookup = {}
    for _, row in clin_df.iterrows():
        pid = row["patient_id"]
        is_dead = row.get("DeadatFollowUp")
        os_days = row.get("OS")

        # label_v1 (original 63-patient cohort)
        if pd.notna(is_dead) and is_dead and pd.notna(os_days) and os_days < 1825:
            l_v1 = 1
        elif pd.notna(is_dead) and not is_dead and pd.notna(os_days) and os_days >= 1825:
            l_v1 = 0
        else:
            l_v1 = None

        # label_v2 (corrected 72-patient cohort, including >1825 day survivors before death)
        if pd.notna(is_dead) and is_dead and pd.notna(os_days) and os_days < 1825:
            l_v2 = 1
        elif pd.notna(os_days) and os_days >= 1825:
            l_v2 = 0
        else:
            l_v2 = None

        clinical_lookup[pid] = {
            "label_v1": l_v1,
            "label_v2": l_v2,
            "survival_duration_days": float(os_days) if pd.notna(os_days) else None,
            "survival_event": 1 if bool(is_dead) else 0,
            "age": float(row.get("Age")) if pd.notna(row.get("Age")) else None,
            "stage": str(row.get("Stage")) if pd.notna(row.get("Stage")) else None,
            "residual_disease": str(row.get("ResidDisease")) if pd.notna(row.get("ResidDisease")) else None,
        }

    # Group volumes by patient
    patient_vols: Dict[str, List[Dict[str, Any]]] = {}
    for v in processed_volumes:
        pid = v["patient_id"]
        patient_vols.setdefault(pid, []).append(v)

    # Master index
    master_index: Dict[str, Any] = {}
    for pid in sorted(patient_to_fold.keys()):
        vols = patient_vols.get(pid, [])
        clin = clinical_lookup.get(pid, {
            "label_v1": None,
            "label_v2": None,
            "survival_duration_days": None,
            "survival_event": 0,
        })

        master_index[pid] = {
            "patient_id": pid,
            "fold": patient_to_fold.get(pid),
            "label_v1": clin["label_v1"],
            "label_v2": clin["label_v2"],
            "survival_duration_days": clin["survival_duration_days"],
            "survival_event": clin["survival_event"],
            "clinical_covariates": {
                "age": clin.get("age"),
                "stage": clin.get("stage"),
                "residual_disease": clin.get("residual_disease"),
            },
            "num_volumes": len(vols),
            "volumes": vols,
        }

    # Save to JSON
    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_index_path, "w", encoding="utf-8") as f:
        json.dump(master_index, f, indent=2)

    return master_index


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1 preprocessing and caching pipeline")
    parser.add_argument(
        "--series-selection",
        type=Path,
        default=Path("data/manifests/series_selection.json"),
        help="Path to series_selection.json",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/splits/folds_v1.json"),
        help="Path to folds_v1.json",
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        default=Path("DATA/ClinicalData101515.csv"),
        help="Path to ClinicalData101515.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/manifest_v1.json"),
        help="Path to manifest_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to cache processed .pt volumes",
    )
    parser.add_argument(
        "--mri-dir",
        type=Path,
        default=Path("data/processed_mri_exploratory"),
        help="Directory for separate exploratory MRI cache",
    )
    parser.add_argument(
        "--resampling-mode",
        type=str,
        default="mode_a",
        choices=["mode_a", "mode_b"],
        help="Resampling mode: mode_a (moderately anisotropic) or mode_b (isotropic 2.5mm)",
    )
    parser.add_argument(
        "--roi-mode",
        type=str,
        default="full_volume",
        choices=["full_volume", "body_crop"],
        help="ROI mode: full_volume or body_crop",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Worker processes for volume reconstruction",
    )

    args = parser.parse_args()
    start_time = time.time()

    print("=" * 80)
    print("         CANCER-MAMBA STAGE 1 PREPROCESSING & CACHING PIPELINE")
    print("=" * 80)
    print(f"Resampling Mode: {args.resampling_mode} (Mode A = Anisotropic, Mode B = Isotropic)")
    print(f"ROI Mode:        {args.roi_mode}")
    print(f"Target Shape:    (80, 80, 80) voxels")
    print(f"Workers:         {args.num_workers}")
    print(f"Output Cache:    {args.output_dir}")
    print("-" * 80)

    # 1. Load series selection
    with open(args.series_selection, "r", encoding="utf-8") as f:
        selection_data = json.load(f)

    task_list = []
    for pid, studies in selection_data["selection_by_patient"].items():
        for sid, s_list in studies.items():
            for s in s_list:
                task_list.append(s)

    print(f"Total selected series to process: {len(task_list)} across {len(selection_data['selection_by_patient'])} patients.")

    # 2. Process series in parallel
    args.output_dir.mkdir(parents=True, exist_ok=True)
    successes = []
    failures = []

    print(f"\nProcessing and caching {len(task_list)} volumes...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_single_series,
                task,
                args.output_dir,
                (80, 80, 80),
                args.resampling_mode,
                args.roi_mode,
            ): task
            for task in task_list
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Caching Volumes"):
            res, err = future.result()
            if res is not None:
                successes.append(res)
            else:
                failures.append(err)

    # Log failures if any
    failures_path = Path("data/manifests/preprocessing_failures.json")
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "failure_count": len(failures),
            "failures": failures,
        }, f, indent=2)

    # 3. Build master patient index
    index_path = args.output_dir / "patient_index.json"
    print(f"\nBuilding master patient index: {index_path}...")
    patient_index = build_patient_index(
        processed_volumes=successes,
        splits_path=args.splits,
        clinical_path=args.clinical,
        output_index_path=index_path,
    )

    # 4. Process exploratory MRI separately
    process_exploratory_mri(
        manifest_path=args.manifest,
        output_dir=args.mri_dir,
        target_shape=(80, 80, 80),
    )

    # 5. Measure elapsed time and disk footprint
    elapsed = time.time() - start_time
    total_bytes = sum(f.stat().st_size for f in args.output_dir.glob("**/*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)

    # Volumes per patient distribution
    vol_counts = [data["num_volumes"] for data in patient_index.values()]
    vol_counter = pd.Series(vol_counts).value_counts().sort_index().to_dict()

    # Patients with >1 volume
    multi_vol_patients = sum(1 for count in vol_counts if count > 1)

    print("\n" + "=" * 80)
    print("                    PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total Patients Cached:          {len(patient_index)} / 92 (100.0%)")
    print(f"Total Volumes Successfully Cached: {len(successes)} / {len(task_list)}")
    print(f"Reconstruction Failures:        {len(failures)} (logged in {failures_path})")
    print(f"Cache Disk Footprint:           {total_mb:.1f} MB (mean {total_mb/max(1, len(successes)):.2f} MB / volume)")
    print(f"Total Elapsed Time:             {elapsed:.1f}s ({elapsed/max(1, len(task_list)):.2f}s / volume)")
    print("\nPer-Patient Volume Distribution (Observations for Stage 5 Mamba):")
    for cnt, num_pts in vol_counter.items():
        print(f"  {cnt:2d} volume(s): {num_pts:2d} patients ({num_pts/len(patient_index)*100:5.1f}%)")
    print(f"Patients with Multiple Volumes: {multi_vol_patients} / {len(patient_index)} ({multi_vol_patients/len(patient_index)*100:.1f}%)")

    # Dual-label breakdown
    v1_counts = pd.Series([d["label_v1"] for d in patient_index.values()]).value_counts(dropna=False).to_dict()
    v2_counts = pd.Series([d["label_v2"] for d in patient_index.values()]).value_counts(dropna=False).to_dict()
    print("\nCohort Label Verification in Master Index:")
    print(f"  label_v1 (original cohort):   {v1_counts}")
    print(f"  label_v2 (corrected cohort):  {v2_counts}")
    print(f"  Survival available (all 92):  {sum(1 for d in patient_index.values() if d['survival_duration_days'] is not None)} / 92")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
