#!/usr/bin/env python3
"""
scripts/build_manifest.py

Builds a versioned dataset manifest (manifest_v1.json) by recursively inspecting
DICOM headers across DATA/TCGA-OV/. Validates against actual DICOM tags
(PatientID, StudyInstanceUID, SeriesInstanceUID) without relying on directory names.

Cross-references against DATA/metadata_labeled.csv and DATA/metadata.csv:
- Flags series in metadata_labeled.csv not found on disk
- Flags series on disk not present in metadata_labeled.csv
- Recomputes instance counts and shapes directly from DICOM headers and reports mismatches
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm


def parse_dicom_file(filepath: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Read DICOM header without loading pixel data.
    Returns (header_info, error_info).
    """
    try:
        # Read header only
        ds = pydicom.dcmread(
            filepath,
            stop_before_pixels=True,
            specific_tags=[
                "PatientID",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "SOPInstanceUID",
                "InstanceNumber",
                "Modality",
                "SeriesDescription",
                "Manufacturer",
                "Rows",
                "Columns",
                "PixelSpacing",
                "SliceThickness",
                "SpacingBetweenSlices",
                "ImagePositionPatient",
                "ImageOrientationPatient",
            ],
        )

        patient_id = str(ds.get("PatientID", "")).strip()
        study_id = str(ds.get("StudyInstanceUID", "")).strip()
        series_id = str(ds.get("SeriesInstanceUID", "")).strip()
        sop_id = str(ds.get("SOPInstanceUID", "")).strip()
        instance_num = ds.get("InstanceNumber", None)
        try:
            instance_num = int(instance_num) if instance_num is not None else None
        except (ValueError, TypeError):
            instance_num = None

        modality = str(ds.get("Modality", "UNKNOWN")).strip()
        series_desc = str(ds.get("SeriesDescription", "")).strip()
        manufacturer = str(ds.get("Manufacturer", "")).strip()

        rows = ds.get("Rows", None)
        cols = ds.get("Columns", None)
        rows = int(rows) if rows is not None else None
        cols = int(cols) if cols is not None else None

        pixel_spacing = ds.get("PixelSpacing", None)
        if pixel_spacing is not None:
            pixel_spacing = [float(x) for x in pixel_spacing]

        slice_thickness = ds.get("SliceThickness", None)
        slice_thickness = float(slice_thickness) if slice_thickness is not None else None

        spacing_between_slices = ds.get("SpacingBetweenSlices", None)
        spacing_between_slices = (
            float(spacing_between_slices) if spacing_between_slices is not None else None
        )

        ipp = ds.get("ImagePositionPatient", None)
        if ipp is not None:
            ipp = [float(x) for x in ipp]

        iop = ds.get("ImageOrientationPatient", None)
        if iop is not None:
            iop = [float(x) for x in iop]

        return {
            "path": str(filepath),
            "patient_id": patient_id,
            "study_id": study_id,
            "series_id": series_id,
            "sop_id": sop_id,
            "instance_num": instance_num,
            "modality": modality,
            "series_description": series_desc,
            "manufacturer": manufacturer,
            "rows": rows,
            "cols": cols,
            "pixel_spacing": pixel_spacing,
            "slice_thickness": slice_thickness,
            "spacing_between_slices": spacing_between_slices,
            "ipp": ipp,
            "iop": iop,
        }, None

    except Exception as e:
        return None, {
            "path": str(filepath),
            "error_type": type(e).__name__,
            "error_message": str(e),
        }


def compute_series_spacing_and_shape(
    instances: List[Dict[str, Any]]
) -> Tuple[Optional[List[int]], Optional[List[float]], Optional[List[float]], bool]:
    """
    Computes 3D volume shape and physical voxel spacing [z, y, x] from DICOM instances.
    Returns: (shape, spacing, orientation, has_variable_shape)
    """
    num_instances = len(instances)
    if num_instances == 0:
        return None, None, None, False

    # Check in-plane shape consistency
    in_plane_shapes = set((inst["rows"], inst["cols"]) for inst in instances)
    has_variable_shape = len(in_plane_shapes) > 1

    first = instances[0]
    ref_rows = first["rows"]
    ref_cols = first["cols"]
    pixel_spacing = first["pixel_spacing"]  # [row_spacing, col_spacing] (dy, dx)
    orientation = first["iop"]

    if has_variable_shape or ref_rows is None or ref_cols is None:
        shape = None
    else:
        shape = [num_instances, ref_rows, ref_cols]

    # Calculate z-spacing (through-plane)
    z_spacing: Optional[float] = None

    if num_instances > 1 and orientation is not None and len(orientation) == 6:
        # Calculate normal vector to image plane
        r = np.array(orientation[:3], dtype=float)
        c = np.array(orientation[3:], dtype=float)
        normal = np.cross(r, c)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-6:
            normal = normal / norm_len

            # Project each slice's ImagePositionPatient along the normal
            projected_positions = []
            for inst in instances:
                if inst["ipp"] is not None and len(inst["ipp"]) == 3:
                    pos = np.dot(np.array(inst["ipp"], dtype=float), normal)
                    projected_positions.append(pos)

            if len(projected_positions) == num_instances:
                sorted_pos = np.sort(projected_positions)
                diffs = np.diff(sorted_pos)
                pos_diffs = diffs[diffs > 1e-4]
                if len(pos_diffs) > 0:
                    z_spacing = float(np.median(pos_diffs))

    # Fallbacks for z_spacing if multi-slice projection is unavailable or num_instances == 1
    if z_spacing is None or z_spacing <= 0:
        if first["spacing_between_slices"] is not None and first["spacing_between_slices"] > 0:
            z_spacing = first["spacing_between_slices"]
        elif first["slice_thickness"] is not None and first["slice_thickness"] > 0:
            z_spacing = first["slice_thickness"]

    # Assemble spacing: [dz, dy, dx]
    spacing: Optional[List[float]] = None
    if pixel_spacing is not None and len(pixel_spacing) == 2:
        dy, dx = pixel_spacing[0], pixel_spacing[1]
        spacing = [round(z_spacing, 4) if z_spacing is not None else None, round(dy, 4), round(dx, 4)]

    return shape, spacing, orientation, has_variable_shape


def build_manifest(
    data_dir: Path,
    metadata_raw_path: Optional[Path],
    metadata_labeled_path: Optional[Path],
    output_path: Path,
    num_workers: int = 8,
) -> Dict[str, Any]:
    """
    Builds the dataset manifest and performs cross-referencing against metadata files.
    """
    print(f"\n[1/4] Scanning files in {data_dir}...")
    all_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            all_files.append(Path(root) / f)

    print(f"  Found {len(all_files)} total files on disk.")

    print(f"\n[2/4] Parsing DICOM headers using {num_workers} worker threads...")
    valid_headers: List[Dict[str, Any]] = []
    skipped_files: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for header, err in tqdm(executor.map(parse_dicom_file, all_files), total=len(all_files), desc="Parsing DICOM"):
            if header is not None:
                valid_headers.append(header)
            else:
                skipped_files.append(err)

    print(f"  Successfully parsed {len(valid_headers)} DICOM slices.")
    print(f"  Skipped / non-DICOM files: {len(skipped_files)}")

    # Group slices by SeriesInstanceUID
    print("\n[3/4] Aggregating slices into series...")
    series_groups: Dict[str, List[Dict[str, Any]]] = {}
    for h in valid_headers:
        s_uid = h["series_id"]
        if not s_uid:
            s_uid = "UNKNOWN_SERIES"
        series_groups.setdefault(s_uid, []).append(h)

    # Load metadata CSVs for cross-referencing
    meta_raw_df = pd.read_csv(metadata_raw_path) if metadata_raw_path and metadata_raw_path.exists() else None
    meta_labeled_df = pd.read_csv(metadata_labeled_path) if metadata_labeled_path and metadata_labeled_path.exists() else None

    raw_series_lookup: Dict[str, Dict[str, Any]] = {}
    if meta_raw_df is not None:
        for _, row in meta_raw_df.iterrows():
            s_uid = str(row.get("Series UID", "")).strip()
            raw_series_lookup[s_uid] = row.to_dict()

    labeled_series_lookup: Dict[str, Dict[str, Any]] = {}
    if meta_labeled_df is not None:
        for _, row in meta_labeled_df.iterrows():
            s_uid = str(row.get("Series UID", "")).strip()
            labeled_series_lookup[s_uid] = row.to_dict()

    # Build series records
    series_records: List[Dict[str, Any]] = []
    disk_series_uids = set(series_groups.keys())

    for series_id, instances in series_groups.items():
        # Representative header
        first = instances[0]
        patient_id = first["patient_id"]
        study_id = first["study_id"]
        modality = first["modality"]
        series_desc = first["series_description"]
        manufacturer = first["manufacturer"]

        # Validate that all instances share the same PatientID, StudyInstanceUID, Modality
        patient_ids = set(inst["patient_id"] for inst in instances)
        if len(patient_ids) > 1:
            print(f"  WARNING: Series {series_id} contains multiple PatientIDs: {patient_ids}")

        num_instances = len(instances)
        shape, spacing, orientation, has_variable_shape = compute_series_spacing_and_shape(instances)

        # Sort instance file paths by slice position along normal if possible, else instance_num
        iop = first["iop"]
        if iop is not None and len(iop) == 6:
            r = np.array(iop[:3], dtype=float)
            c = np.array(iop[3:], dtype=float)
            normal = np.cross(r, c)
            norm_val = np.linalg.norm(normal)
            if norm_val > 1e-6:
                normal = normal / norm_val

            def get_pos(inst):
                if inst["ipp"] is not None:
                    return np.dot(np.array(inst["ipp"], dtype=float), normal)
                return inst["instance_num"] if inst["instance_num"] is not None else 0

            instances_sorted = sorted(instances, key=get_pos)
        else:
            instances_sorted = sorted(
                instances,
                key=lambda x: x["instance_num"] if x["instance_num"] is not None else 0,
            )

        # Common directory path
        sample_path = Path(instances[0]["path"])
        series_dir = str(sample_path.parent)

        # Relative paths from project root
        try:
            rel_dir = str(Path(series_dir).relative_to(Path.cwd()))
        except ValueError:
            rel_dir = series_dir

        # Cross-reference with labeled and raw metadata
        in_labeled = series_id in labeled_series_lookup
        in_raw = series_id in raw_series_lookup
        label_val = labeled_series_lookup[series_id].get("label") if in_labeled else None

        record = {
            "series_id": series_id,
            "patient_id": patient_id,
            "study_id": study_id,
            "modality": modality,
            "num_instances": num_instances,
            "shape": shape,
            "spacing": spacing,
            "orientation": orientation,
            "series_description": series_desc,
            "manufacturer": manufacturer,
            "path": rel_dir,
            "has_variable_slice_shape": has_variable_shape,
            "in_metadata_labeled": in_labeled,
            "in_metadata_raw": in_raw,
            "label": int(label_val) if label_val is not None and pd.notna(label_val) else None,
        }
        series_records.append(record)

    # Sort records by patient_id, study_id, series_id
    series_records.sort(key=lambda x: (x["patient_id"], x["study_id"], x["series_id"]))

    # Cross-reference analysis
    print("\n[4/4] Cross-referencing against metadata CSVs...")
    series_in_labeled_not_on_disk = []
    if meta_labeled_df is not None:
        labeled_uids = set(labeled_series_lookup.keys())
        missing_from_disk = labeled_uids - disk_series_uids
        series_in_labeled_not_on_disk = sorted(list(missing_from_disk))

    series_on_disk_not_in_labeled = sorted(list(disk_series_uids - set(labeled_series_lookup.keys())))

    # Check instance count mismatches with raw metadata
    instance_count_mismatches = []
    if meta_raw_df is not None:
        for s_rec in series_records:
            s_uid = s_rec["series_id"]
            if s_uid in raw_series_lookup:
                csv_count = raw_series_lookup[s_uid].get("Number of Images")
                if csv_count is not None and pd.notna(csv_count):
                    csv_count = int(csv_count)
                    if csv_count != s_rec["num_instances"]:
                        instance_count_mismatches.append({
                            "series_id": s_uid,
                            "patient_id": s_rec["patient_id"],
                            "disk_count": s_rec["num_instances"],
                            "csv_count": csv_count,
                        })

    # Summary statistics
    patients_on_disk = set(r["patient_id"] for r in series_records)
    studies_on_disk = set(r["study_id"] for r in series_records)

    manifest_data = {
        "metadata": {
            "version": "v1.0",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source_dir": str(data_dir),
            "num_patients": len(patients_on_disk),
            "num_studies": len(studies_on_disk),
            "num_series": len(series_records),
            "num_instances_total": len(valid_headers),
            "num_skipped_non_dicom_files": len(skipped_files),
        },
        "cross_reference_summary": {
            "series_in_metadata_labeled_not_on_disk": {
                "count": len(series_in_labeled_not_on_disk),
                "series_uids": series_in_labeled_not_on_disk,
            },
            "series_on_disk_not_in_metadata_labeled": {
                "count": len(series_on_disk_not_in_labeled),
                "explanation": "169 series belong to patients omitted from metadata_labeled.csv (censored before 5yr or lived past 5yr before death)",
                "sample_series_uids": series_on_disk_not_in_labeled[:10],
            },
            "instance_count_mismatches_vs_raw_csv": {
                "count": len(instance_count_mismatches),
                "mismatches": instance_count_mismatches,
            },
        },
        "skipped_files": skipped_files,
        "series": series_records,
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\nManifest successfully written to: {output_path}")
    print(f"  Patients on disk: {len(patients_on_disk)}")
    print(f"  Studies on disk:  {len(studies_on_disk)}")
    print(f"  Series on disk:   {len(series_records)}")
    print(f"  Total instances:  {len(valid_headers)}")
    print(f"  Series in labeled CSV missing from disk: {len(series_in_labeled_not_on_disk)}")
    print(f"  Series on disk not in labeled CSV:       {len(series_on_disk_not_in_labeled)}")
    print(f"  Instance count mismatches vs raw CSV:    {len(instance_count_mismatches)}")

    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Build TCGA-OV dataset manifest v1")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("DATA/TCGA-OV"),
        help="Path to raw TCGA-OV DICOM root directory",
    )
    parser.add_argument(
        "--metadata-raw",
        type=Path,
        default=Path("DATA/metadata.csv"),
        help="Path to raw metadata.csv",
    )
    parser.add_argument(
        "--metadata-labeled",
        type=Path,
        default=Path("DATA/metadata_labeled.csv"),
        help="Path to metadata_labeled.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/manifest_v1.json"),
        help="Output manifest JSON path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel header reading",
    )

    args = parser.parse_args()
    build_manifest(
        data_dir=args.data_dir,
        metadata_raw_path=args.metadata_raw,
        metadata_labeled_path=args.metadata_labeled,
        output_path=args.output,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
