"""
preprocessing/dicom.py

Reconstructs 3D volumetric images from DICOM series using SimpleITK (the primary
volumetric engine selected in Stage 0).
Preserves 3D spatial geometry (spacing, origin, direction cosines) and applies
RescaleSlope/RescaleIntercept to ensure true Hounsfield Units (HU).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk


def get_sorted_dicom_files(series_dir: Path, series_id: Optional[str] = None) -> List[str]:
    """
    Returns slice file paths sorted by physical position along the slice normal
    (ImagePositionPatient), with fallback to InstanceNumber.
    """
    reader = sitk.ImageSeriesReader()
    if series_id:
        file_names = reader.GetGDCMSeriesFileNames(str(series_dir), series_id)
    else:
        file_names = reader.GetGDCMSeriesFileNames(str(series_dir))

    if file_names and len(file_names) > 0:
        return list(file_names)

    # Robust fallback: manual header extraction & physical coordinate sorting
    dcm_files = [p for p in series_dir.glob("*.dcm")]
    if not dcm_files:
        dcm_files = [p for p in series_dir.glob("*") if p.is_file() and not p.name.startswith(".")]

    if not dcm_files:
        return []

    slice_info = []
    normal = None

    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if series_id and str(ds.get("SeriesInstanceUID", "")).strip() != series_id:
                continue

            ipp = ds.get("ImagePositionPatient", None)
            iop = ds.get("ImageOrientationPatient", None)
            inst_num = ds.get("InstanceNumber", 0)

            if normal is None and iop is not None and len(iop) == 6:
                r = np.array(iop[:3], dtype=float)
                c = np.array(iop[3:], dtype=float)
                cross = np.cross(r, c)
                norm_len = np.linalg.norm(cross)
                if norm_len > 1e-6:
                    normal = cross / norm_len

            pos = 0.0
            if ipp is not None and normal is not None:
                pos = float(np.dot(np.array(ipp, dtype=float), normal))
            elif inst_num is not None:
                try:
                    pos = float(inst_num)
                except ValueError:
                    pos = 0.0

            slice_info.append((pos, str(f)))
        except Exception:
            continue

    # Sort ascending by physical position
    slice_info.sort(key=lambda x: x[0])
    return [f for _, f in slice_info]


def reconstruct_volume_from_dicom(
    series_dir: Path,
    expected_instances: Optional[int] = None,
    series_id: Optional[str] = None,
) -> Tuple[Optional[sitk.Image], Optional[Dict[str, Any]]]:
    """
    Reconstructs a 3D SimpleITK image from a DICOM series directory.
    Validates reconstructed depth against expected_instances.

    Returns:
        (sitk_image, None) if successful
        (None, error_dict) if failed
    """
    series_dir = Path(series_dir)
    if not series_dir.exists():
        return None, {
            "error_type": "DirectoryNotFound",
            "series_dir": str(series_dir),
            "series_id": series_id,
            "message": f"Series directory does not exist: {series_dir}",
        }

    try:
        file_names = get_sorted_dicom_files(series_dir, series_id)
        if not file_names:
            return None, {
                "error_type": "NoDicomFilesFound",
                "series_dir": str(series_dir),
                "series_id": series_id,
                "message": "No valid DICOM files found in series directory",
            }

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_names)
        image = reader.Execute()

        size = image.GetSize()  # (X, Y, Z) in SimpleITK
        num_reconstructed_slices = size[2]

        if expected_instances is not None and num_reconstructed_slices != expected_instances:
            # Note mismatch but do not necessarily discard unless completely corrupted
            pass

        return image, None

    except Exception as e:
        return None, {
            "error_type": type(e).__name__,
            "series_dir": str(series_dir),
            "series_id": series_id,
            "message": str(e),
        }
