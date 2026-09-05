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

---

## 6. Stage 1: Preprocessing, Standardization & Caching

### Resampling Strategy (Mode A vs Mode B)
Stage 0 revealed severe physical voxel anisotropy in CT series (median z-spacing 5.0mm vs xy-spacing 0.75mm; 6.8:1 ratio).
- **Mode A (Default Baseline)**: Resamples to a moderately anisotropic target spacing (`1.5mm x 1.5mm x 3.0mm`), then crops/pads to an `80x80x80` voxel grid. This respects native physical slice thickness without manufacturing phantom slices via extreme 1mm interpolation.
- **Mode B (Ablation)**: Full isotropic resampling to `2.5mm³` followed by crop/pad to `80x80x80`.

### HU Windowing & Intensity Normalization
- Clipped to standard abdominal soft-tissue window: `[-135, 215]` HU (accounting for 89.2% of non-air voxels).
- Normalized to `[0.0, 1.0]`.

### Multi-Phase Observation Preservation
Rather than arbitrarily picking a single volume per patient, studies with multiple contrast phases (e.g. non-contrast, arterial, portal venous, delayed) are cached as separate volumes (`203` total volumes across `92` patients; `64.1%` of patients have multiple observations). This preserves distinct anatomical/functional observations for the patient-level Mamba sequence modeling in Stage 5. Duplicate reconstructions within the same phase are tie-broken by selecting the one with the maximum slice count.

### Patient-Level 5-Fold Cross-Validation (`data/splits/folds_v1.json`)
- Split strictly at the `patient_id` level (zero data leakage).
- Stratified on `label_v2` (45:27 cohort; exactly 9 class 1 and 5-6 class 0 per fold).
- 20 unlabeled survival patients evenly distributed (4 per fold).
- No static held-out test set initially: given the small cohort (92 patients), reserving 15-20% static test data reduces training sample size excessively; performance is assessed via nested out-of-fold cross-validation.

### Running Stage 1 Pipeline
```bash
# 1. Primary series selection
python preprocessing/series_selection.py

# 2. Generate 5-fold splits & run unit tests
python scripts/create_splits.py
pytest tests/test_splits.py -v

# 3. End-to-end preprocessing & caching
python scripts/preprocess.py --num-workers 6
```

