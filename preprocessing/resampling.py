"""
preprocessing/resampling.py

Spatial standardization and resampling routines for 3D volumetric CT.
Supports:
  - Resampling Mode A (default): Moderately anisotropic target spacing (1.5mm x 1.5mm x 3.0mm)
    respecting native slice thickness without manufacturing phantom slices.
  - Resampling Mode B (ablation): Full isotropic resampling to 2.5mm^3.
  - ROI Mode A (default): Full standardized volume (center crop/pad).
  - ROI Mode B: Body-region crop via thresholding (> -500 HU) to remove surrounding air.
"""

from typing import Optional, Tuple, Union
import numpy as np
import SimpleITK as sitk
import torch


def extract_body_bounding_box(
    arr_hu: np.ndarray,
    air_threshold: float = -500.0,
    margin: int = 2,
) -> Tuple[slice, slice, slice]:
    """
    Finds the 3D bounding box containing non-air voxels (HU > air_threshold).
    arr_hu: (Z, Y, X) numpy array.
    Returns: (slice_z, slice_y, slice_x)
    """
    body_mask = arr_hu > air_threshold
    if not np.any(body_mask):
        return slice(0, arr_hu.shape[0]), slice(0, arr_hu.shape[1]), slice(0, arr_hu.shape[2])

    z_indices = np.where(body_mask.any(axis=(1, 2)))[0]
    y_indices = np.where(body_mask.any(axis=(0, 2)))[0]
    x_indices = np.where(body_mask.any(axis=(0, 1)))[0]

    z_min = max(0, int(z_indices[0]) - margin)
    z_max = min(arr_hu.shape[0], int(z_indices[-1]) + 1 + margin)

    y_min = max(0, int(y_indices[0]) - margin)
    y_max = min(arr_hu.shape[1], int(y_indices[-1]) + 1 + margin)

    x_min = max(0, int(x_indices[0]) - margin)
    x_max = min(arr_hu.shape[2], int(x_indices[-1]) + 1 + margin)

    return slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max)


def crop_or_pad_3d(
    arr: np.ndarray,
    target_shape: Tuple[int, int, int] = (80, 80, 80),
    pad_val: float = 0.0,
) -> np.ndarray:
    """
    Centers the 3D array in the target_shape (Z, Y, X) by cropping or padding.
    """
    out = np.full(target_shape, pad_val, dtype=arr.dtype)
    z_in, y_in, x_in = arr.shape
    z_out, y_out, x_out = target_shape

    # Z-axis
    z_src_start = max(0, (z_in - z_out) // 2)
    z_src_end = min(z_in, z_src_start + z_out)
    z_dst_start = max(0, (z_out - z_in) // 2)
    z_dst_end = z_dst_start + (z_src_end - z_src_start)

    # Y-axis
    y_src_start = max(0, (y_in - y_out) // 2)
    y_src_end = min(y_in, y_src_start + y_out)
    y_dst_start = max(0, (y_out - y_in) // 2)
    y_dst_end = y_dst_start + (y_src_end - y_src_start)

    # X-axis
    x_src_start = max(0, (x_in - x_out) // 2)
    x_src_end = min(x_in, x_src_start + x_out)
    x_dst_start = max(0, (x_out - x_in) // 2)
    x_dst_end = x_dst_start + (x_src_end - x_src_start)

    out[z_dst_start:z_dst_end, y_dst_start:y_dst_end, x_dst_start:x_dst_end] = arr[
        z_src_start:z_src_end, y_src_start:y_src_end, x_src_start:x_src_end
    ]
    return out


def resample_image_sitk(
    image: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.5, 1.5, 3.0),
    default_value: float = -1024.0,
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    """
    Resamples a SimpleITK image to the given target_spacing (X, Y, Z in mm).
    """
    orig_spacing = image.GetSpacing()  # (x, y, z)
    orig_size = image.GetSize()        # (x, y, z)

    # Calculate new dimensions
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]
    # Ensure minimum 1 in all dimensions
    new_size = [max(1, s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetDefaultPixelValue(default_value)

    return resample.Execute(image)


def standardize_volume(
    image: sitk.Image,
    target_shape: Tuple[int, int, int] = (80, 80, 80),
    resampling_mode: str = "mode_a",
    roi_mode: str = "full_volume",
    min_hu: float = -135.0,
    max_hu: float = 215.0,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Complete volume standardization pipeline:
      1. Optional Body ROI cropping (if roi_mode == 'body_crop')
      2. Physical resampling (Mode A anisotropic or Mode B isotropic)
      3. HU windowing and normalization to [0, 1]
      4. Crop/pad to target voxel grid (Z, Y, X)

    Returns:
        standardized_array: np.ndarray of shape target_shape (Z, Y, X), values in [0.0, 1.0]
        actual_spacing: Tuple of (spacing_x, spacing_y, spacing_z) in mm
    """
    # Define physical spacing based on mode
    # SimpleITK spacing is (X, Y, Z)
    if resampling_mode.lower() == "mode_b":
        # Mode B: Isotropic 2.5mm^3 ablation
        target_spacing = (2.5, 2.5, 2.5)
    else:
        # Mode A (Default): Moderately anisotropic (1.5mm x 1.5mm in-plane, 3.0mm through-plane)
        target_spacing = (1.5, 1.5, 3.0)

    # Handle ROI mode
    if roi_mode.lower() == "body_crop":
        arr_raw = sitk.GetArrayFromImage(image)  # (Z, Y, X)
        z_sl, y_sl, x_sl = extract_body_bounding_box(arr_raw, air_threshold=-500.0)

        # Crop SimpleITK image using voxel indices: [startX, startY, startZ], [sizeX, sizeY, sizeZ]
        start_index = [x_sl.start, y_sl.start, z_sl.start]
        size_crop = [x_sl.stop - x_sl.start, y_sl.stop - y_sl.start, z_sl.stop - z_sl.start]

        # Verify positive sizes
        if all(s > 0 for s in size_crop):
            image = sitk.RegionOfInterest(image, size_crop, start_index)

    # Resample image
    resampled_img = resample_image_sitk(image, target_spacing=target_spacing, default_value=-1024.0)
    actual_spacing = resampled_img.GetSpacing()

    # Convert to numpy (Z, Y, X)
    arr_hu = sitk.GetArrayFromImage(resampled_img)

    # HU windowing & normalization to [0.0, 1.0]
    clipped = np.clip(arr_hu, min_hu, max_hu).astype(np.float32)
    normalized = (clipped - min_hu) / (max_hu - min_hu)

    # Crop or pad to target shape (Z, Y, X)
    # Background padding value is 0.0 (corresponds to min_hu = -135 HU or air)
    final_vol = crop_or_pad_3d(normalized, target_shape=target_shape, pad_val=0.0)

    return final_vol, actual_spacing
