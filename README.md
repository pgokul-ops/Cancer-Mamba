# Cancer-Mamba: Hierarchical Mamba Pipeline for Patient-Level Volumetric Cancer Imaging (TCGA-OV)

A deep learning framework for patient-level volumetric cancer imaging on the TCGA-OV (Ovarian Serous Cystadenocarcinoma) cohort.

---

## 1. Hardware Specification & Constraints

- **Target GPU**: NVIDIA GeForce GTX 1650 (Laptop/Desktop)
- **Architecture**: Turing (`sm_75`, Compute Capability 7.5)
- **VRAM**: 4096 MiB (4 GB)
- **Driver Version**: 580.173.02 (supports CUDA up to 13.0)
- **Host OS**: Linux x86_64

### 4GB VRAM Constraints
Due to the strict 4GB VRAM budget:
- Training must utilize memory-efficient strategies (mixed precision, gradient checkpointing, small candidate volumes such as 64³–96³).
- Pipeline relies on optimized data loading and preprocessed cached volumes rather than on-the-fly heavy DICOM loading during training.

---

## 2. Environment Setup

### Prerequisites
- Python 3.10 or 3.11
- PyTorch with CUDA support matching your driver (CUDA 11.8 / 12.x / 13.x)

```bash
# Recommended: create a virtual or conda environment
conda create -n cancer python=3.10 -y
conda activate cancer

# Install dependencies via pip
pip install -e .
```

### Mamba-SSM on Turing Architecture (`sm_75`)
> **Important Note on `mamba-ssm`**:
> Pre-built wheels for `mamba-ssm` and `causal-conv1d` distributed on PyPI officially target Ampere (`sm_80`), Ada (`sm_89`), and Hopper (`sm_90`).
> The GTX 1650 has compute capability **7.5** (`sm_75`). While the underlying CUDA code can support `sm_75`, running hardware-accelerated Mamba on Turing requires building from source with `nvcc`:
> ```bash
> export TORCH_CUDA_ARCH_LIST="7.5"
> pip install --no-build-isolation causal-conv1d>=1.2.0
> pip install --no-build-isolation mamba-ssm>=1.2.0
> ```
> For Stages 0–2, we do **not** force-install `mamba-ssm`. The pipeline falls back to an equivalent pure PyTorch scan-based SSM implementation, ensuring portability and stability across development environments. Stage 3 will revisit hardware-accelerated kernels.

---

## 3. Primary DICOM->Volume Path Justification

We designate **`SimpleITK`** as the primary DICOM-to-3D-volume reconstruction path (complemented by `pydicom` for fast metadata extraction):
- **Why SimpleITK over manual slice stacking in pydicom**:
  DICOM series slices can be non-consecutive, out-of-order, or acquired with gantry tilt. `SimpleITK.ImageSeriesReader` automatically sorts slices via `ImagePositionPatient` along the slice normal, checks for uniform slice spacing, properly applies Hounsfield Unit (HU) linear rescale (`RescaleSlope` and `RescaleIntercept`), and encapsulates the 3D direction matrix, physical origin, and voxel spacing.
- **Why SimpleITK over nibabel**:
  Nibabel is natively designed for NIfTI/Analyze formats. Its DICOM support requires external wrappers or converters (e.g., `dcm2niix`), whereas SimpleITK natively consumes DICOM series directly.

---

## 4. Repository Skeleton

```
Cancer-Mamba/
├── pyproject.toml              # Build & dependency definitions
├── README.md                   # Project documentation & hardware specifications
├── configs/                    # Experiment & pipeline configurations
├── data/
│   ├── manifests/              # Versioned manifests (manifest_v1.json, dataset_report.json)
│   ├── processed/              # Resampled, normalized volumetric caches (Stage 1+)
│   └── splits/                 # Patient-level train/val/test splits
├── preprocessing/              # Volume extraction, HU windowing, isotropic resampling
├── datasets/                   # PyTorch Dataset and DataLoader implementations
├── models/                     # Hierarchical 3D Mamba / SSM backbones
├── heads/                      # Outcome prediction & survival heads
├── training/                   # Training loop, loss functions, optimizer
├── pretraining/                # Self-supervised pretraining modules
├── evaluation/                 # Validation metrics, ROC-AUC, C-index
├── experiments/                # Experiment tracking and hyperparameter sweeps
└── scripts/
    ├── build_manifest.py       # Stage 0: DICOM header manifest builder
    └── audit_dataset.py        # Stage 0: Cohort & modality auditor
```

---

## 5. Stage 0: Dataset Audit & Manifest Generation

To build the manifest from raw DICOM headers:
```bash
python scripts/build_manifest.py
```

To run the dataset audit and produce `dataset_report.json`:
```bash
python scripts/audit_dataset.py
```
