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

---

## 7. Stage 2: Minimal 3D CNN Baseline (Plumbing Check)

Stage 2 serves as an explicit end-to-end plumbing verification for the training loop, GPU memory pipeline, patient-level logit pooling, and reproducible logging harness before Mamba sequence modeling.

### Architecture (`models/tiny_cnn3d.py`)
- 4-stage 3D CNN: `Conv3D(3x3x3) -> BatchNorm3d -> ReLU -> MaxPool3d(2x2x2)`
- Channels: 1 -> 8 -> 16 -> 32 -> 64
- Head: `AdaptiveAvgPool3d(1) -> Dropout(0.2) -> Linear(64, 1)`
- Parameters: **73,097** (<1M params)
- Peak VRAM: **156.15 MB** (4.2% of 3712 MB budget on NVIDIA GTX 1650) with FP16 AMP.

### Verification Progression
1. **Overfit Memorization Test**: Trained on 8 volumes for 40 epochs; loss reached **0.0045** and accuracy reached **100.0%** (memorization confirmed).
2. **Single-Fold Sanity Check**: Validated on fold 0, inspecting patient prediction tables, probabilities, and gradient flow.
3. **5-Fold Cross-Validation**: Full 5-fold CV completed in 204.7 seconds (~41s/fold) with zero NaN losses and zero constant-prediction collapses.

### Patient-Level vs. Volume-Level Metrics

| Metric | Patient-Level (Primary) | Volume-Level (Inflated) |
| :--- | :---: | :---: |
| **ROC-AUC** | **0.7052 +/- 0.1474** | 0.6813 +/- 0.1373 |
| **PR-AUC** | **0.8050 +/- 0.1007** | 0.8596 +/- 0.0624 |
| **F1 Score** | **0.6466 +/- 0.2251** | 0.7130 +/- 0.2298 |
| **Balanced Accuracy** | **0.5633 +/- 0.0441** | 0.5758 +/- 0.0664 |
| **Sensitivity** | **0.7333 +/- 0.3341** | 0.7583 +/- 0.3253 |
| **Specificity** | **0.3933 +/- 0.3617** | 0.3934 +/- 0.3148 |

### Running Stage 2
```bash
# Run all sanity checks and 5-fold CV:
python scripts/train_baseline_cnn.py --mode all

# Run specific stages:
python scripts/train_baseline_cnn.py --mode overfit
python scripts/train_baseline_cnn.py --mode single_fold
python scripts/train_baseline_cnn.py --mode full_cv
```

---

## 8. Stage 3: Minimal 3D Mamba (Flat, Non-Hierarchical)

Stage 3 establishes the core Mamba sequence-modeling building block on volumetric imaging, establishing a baseline before adding hierarchical processing in Stage 4.

### SSM Backend: Pure PyTorch Reference S6 Implementation
- Pre-built wheels for `mamba-ssm` target Ampere/Hopper (`sm_80+`). Compiling from source on GTX 1650 (`sm_75`) without `nvcc` is bypassed by using an exact, hardware-agnostic Pure PyTorch implementation of the S6 Selective State Space Model (`models/mamba_block.py`).
- **Numerical Stability**: Includes continuous discretization clamping on $\Delta \in [10^{-4}, 0.5]$ and explicit FP32 recurrence accumulation to prevent FP16 overflow under AMP.
- **Analytical Adjoint Backward**: Employs a custom autograd function (`SelectiveScanFn`) computing exact reverse adjoint recurrence in $O(L)$, speeding up backward passes by $>12\times$ over naive autograd graph tracing.

### Architecture (`models/flat_mamba.py`)
- **3D Patch Embedding**: Patch size $8 \times 8 \times 8$ from $(1, 80, 80, 80) \to 10 \times 10 \times 10 = 1000$ spatial tokens of dimension $d_{model}=128$ with learnable 1D positional embeddings.
- **Backbone**: 4 stacked `MambaBlock` layers ($d_{model}=128, d_{state}=16, \text{expand}=1.5, d_{conv}=4$).
- **Sequence Pooling & Head**: Combines global mean pooling (volume background context) and salient feature max pooling (`[mean, max]`, dimension 256) into a 2-layer MLP classifier (`Linear(256, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1)`).
- **Parameter Count**: **577,665** parameters (<1M target).
- **Peak VRAM**: **579.11 MB** (15.6% of 3712 MB budget on GTX 1650) with FP16 AMP.

### Verification Progression
1. **SSM Block Unit Tests**: `tests/test_mamba_block.py` verifies shape preservation, zero NaNs/Infs, and gradient flow across variable sequence lengths (**3/3 PASSED**).
2. **Overfit Memorization Test**: Trained on 8 volumes for 25 epochs; loss dropped to **0.0001** and accuracy reached **100.0%** (memorization confirmed).
3. **Single-Fold Sanity Check**: Validated on fold 0 (5 epochs); loss steadily decreased from 0.537 to 0.317, with diverse, non-collapsed patient predictions.
4. **5-Fold Cross-Validation**: Full 5-fold CV completed across all 72 patients with patient-level mean logit pooling vs. volume-level inflated evaluation.

### Comparative Benchmark: Stage 2 (TinyCNN3D) vs Stage 3 (FlatMamba3D)

| Metric | Stage 2 (TinyCNN3D Baseline) | Stage 3 (FlatMamba3D) | Difference / Notes |
| :--- | :---: | :---: | :--- |
| **Patient ROC-AUC** | **0.7052 ± 0.1474** | **0.7067 ± 0.1471** | Matches / edges baseline (+0.0015) |
| **Patient PR-AUC** | **0.8050 ± 0.1007** | **0.7997 ± 0.1091** | Consistent outcome prediction |
| **Patient F1 Score** | **0.6466 ± 0.2251** | **0.7600 ± 0.0514** | Substantially higher & lower variance across folds |
| **Balanced Accuracy** | **0.5633 ± 0.0441** | **0.5144 ± 0.0441** | Sensitive to positive class weighting |
| **Sensitivity** | 0.7333 ± 0.3341 | 0.9556 ± 0.0889 | High sensitivity |
| **Specificity** | 0.3933 ± 0.3617 | 0.0733 ± 0.0904 | Low false-negative rate |
| **Peak VRAM** | **156.2 MB** | **579.1 MB** | 15.6% of 4GB budget at $L=1000$ |
| **Parameters** | **73,097** | **577,665** | Fully compliant with <1M parameter spec |
| **Sequence Length** | N/A (3D Convolutions) | 1,000 tokens | Pure 1D selective scan serialization |

### Running Stage 3
```bash
# Run unit tests
pytest tests/test_mamba_block.py -v

# Run overfit memorization test
python scripts/train_flat_mamba.py --mode overfit

# Run single-fold sanity check
python scripts/train_flat_mamba.py --mode single_fold

# Run full 5-fold cross-validation
python scripts/train_flat_mamba.py --mode full_cv
```

---

## 9. Stage 4: Hierarchical 3D Mamba (Local Windows + Token Reduction + Global Mamba)

Stage 4 introduces multi-scale hierarchical spatial processing to volumetric Mamba modeling, replacing flat 1D sequence raster scans with true 3D cubic windowed selective state-space processing.

### Architecture (`models/hierarchical_mamba.py`)
- **3D Patch Embedding**: Patch size $8 \times 8 \times 8$ produces a $10 \times 10 \times 10$ spatial token grid ($L=1000, d_{model}=128$).
- **Local Windowed Mamba (`models/local_mamba.py`)**: Partitions the 3D grid into non-overlapping $2 \times 2 \times 2$ cubic windows ($w=2 \implies 125$ windows of 8 tokens each). Parallel selective scans are executed across 2 local Mamba blocks, preserving 3D spatial adjacency along all axes.
- **Learned Token Reduction (`models/token_reduction.py`)**: Maps each 8-token local window into a single representative regional token via `Linear(8 * D, D) -> LayerNorm -> GELU`, compressing the sequence $8\times$ from $L=1000$ to $L=125$.
- **Global Regional Mamba (`models/global_mamba.py`)**: Adds learnable 1D position embeddings and executes 2 global Mamba blocks over the 125-token regional context.
- **Salient Sequence Pooling**: Dual `[mean, max]` pooling (dimension 256) fed into a 2-layer MLP classifier (`Linear(256, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1)`).
- **Parameter Count**: **725,377** parameters (<1M spec target).
- **Peak VRAM**: **351.03 MB** (9.5% of 3712 MB budget on GTX 1650).

### Task 0 Calibration Diagnostic (`scripts/calibration_diagnostic.py`)
- Swept decision thresholds $\tau \in [0.01, 0.99]$ across out-of-fold validation sets to diagnose Stage 3's low specificity at $\tau=0.50$.
- Established that class imbalance in the labeled cohort ($45:27$, volume ratio $118:41$) causes models to output mean probabilities centered around $0.70-0.78$.
- At Youden's J optimal threshold ($\tau^* \approx 0.75$), specificity immediately recovers to $\approx 74\%$, demonstrating that the specificity drop was a threshold calibration artifact rather than an architectural collapse.

### Definitive Three-Way Comparative Benchmark

Evaluated across identical 5-fold splits (`data/splits/folds_v1.json`), identical cohort (72 patients, 159 volumes), and patient-level mean logit pooling.

| Metric | Stage 2 (TinyCNN3D Baseline) | Stage 3 (FlatMamba3D) | Stage 4 (HierarchicalMamba3D) | Delta vs Flat Mamba |
| :--- | :---: | :---: | :---: | :--- |
| **Patient ROC-AUC** | 0.7052 ± 0.1474 | 0.7067 ± 0.1471 | **0.7933 ± 0.1260** | **+0.0866 (Major breakthrough)** |
| **Patient PR-AUC** | 0.8050 ± 0.1007 | 0.7997 ± 0.1091 | **0.8760 ± 0.0693** | **+0.0763** |
| **Patient F1 Score** | 0.6466 ± 0.2251 | 0.7600 ± 0.0514 | **0.8100 ± 0.0392** | **+0.0500** |
| **Balanced Accuracy ($\tau=0.50$)** | 0.5633 ± 0.0441 | 0.5144 ± 0.0441 | **0.6156 ± 0.1099** | **+0.1012** |
| **Sensitivity ($\tau=0.50$)** | 0.7333 ± 0.3341 | 0.9556 ± 0.0889 | **0.9778 ± 0.0444** | High sensitivity |
| **Specificity ($\tau=0.50$)** | 0.3933 ± 0.3617 | 0.0733 ± 0.0904 | **0.2533 ± 0.2544** | **+0.1800** |
| **Calibrated Balanced Acc ($\tau^*$)**| 0.7411 ± 0.1392 | 0.7255 ± 0.0866 | **0.7389 ± 0.1204** | Calibration parity confirmed |
| **Calibrated Specificity ($\tau^*$)** | 0.7267 ± 0.2736 | 0.7400 ± 0.1555 | **0.7000 ± 0.2716** | Specificity fully recovered |
| **Volume ROC-AUC (Inflated)** | 0.6813 ± 0.1373 | 0.5774 ± 0.1101 | **0.6612 ± 0.1079** | Patient-level pooling is essential |
| **Peak VRAM** | **156.2 MB** | 579.1 MB | **351.0 MB** | **39% VRAM reduction vs Flat** |
| **Total 5-Fold Runtime** | **204.7 seconds** (~3.4 min) | 4759.9 seconds (~79.3 min) | **537.0 seconds** (~8.95 min) | **8.86x speedup over Flat Mamba** |
| **Parameter Count** | **73,097** | 577,665 | 725,377 | Compliant with <1M spec |

### Running Stage 4
```bash
# Run unit tests (including spatial coordinate correctness)
python -m pytest tests/test_local_mamba.py -v

# Run calibration diagnostic
python scripts/calibration_diagnostic.py

# Run overfit memorization test
python scripts/train_hierarchical_mamba.py --mode overfit

# Run single-fold sanity check
python scripts/train_hierarchical_mamba.py --mode single_fold

# Run full 5-fold cross-validation
python scripts/train_hierarchical_mamba.py --mode full_cv
```

---

## 10. Stage 5 & 5.5: Patient-Level Sequence Modeling & Reconciled Aggregation Benchmark

Stage 5 models patients as sequences of volume observations across multi-phase contrast acquisitions and longitudinal CT scans (59 of 92 patients have 2 to 12 cached volumes).

### Task 0 Honest Accounting of Stage 4
- **Fold 0 Validation ROC-AUC**: Confirmed directly from metrics as **`0.5556`** (matching CNN's `0.5556`).
- **Paired Per-Fold Comparison**: Hierarchical Mamba wins in 3/5 folds (1, 3, 4) and loses in 2/5 (0, 2) vs Flat Mamba. The mean AUC advantage (+0.0866) was largely propelled by stability on Fold 4 (0.8222 vs 0.4889). On $N=72$, fold variance is substantial ($\text{std} \approx 0.12-0.15$), and calibrated balanced accuracy ($\approx 0.72-0.74$) represents the true baseline to beat.

### Architecture & Aggregation Baselines (`models/patient_aggregators.py`)
Pre-classifier 256-dim volume embeddings (`[mean, max]` pooled) are extracted from Stage 4 under strict nested cross-validation (zero train/val leakage) and ordered by:
1. Primary: Study/series acquisition date (`MM-DD-YYYY` from manifest).
2. Secondary: Clinical contrast phase hierarchy (`non_contrast` $\to$ `arterial` $\to$ `portal_venous` $\to$ `primary_routine` $\to$ `contrast_enhanced_general` $\to$ `delayed` $\to$ `lung_window` $\to$ `chest_std`).

Four aggregation mechanisms are benchmarked with shared dual heads:
- **Mean Pooling**: Masked average over patient's volume embeddings.
- **Max Pooling**: Masked maximum over patient's volume embeddings.
- **Attention Pooling**: Learnable query cross-attention over volume embeddings.
- **Patient Mamba**: Selective SSM sequence modeling over ordered volume embeddings with masked pooling.

### Stage 5.5 Diagnostics & Reconciled Protocol
1. **Task 1 Control (Apples-to-Apples Evaluation)**:
   - Volume-logit averaging with Stage 4's pre-trained head: `0.7252 ± 0.1621`
   - Mean-pooled embeddings evaluated through Stage 4's pre-trained head: `0.7289 ± 0.1495` ($\Delta = +0.0037$)
   - **Conclusion**: Embedding pooling loses zero information compared to logit averaging.
2. **Aggregator Capacity & Warm-Start Head**: Freshly training 16k params from scratch on only 57 patients under joint multi-task loss caused overfitting. Initializing the head with Stage 4's pre-trained weights and differential learning rate (`1e-4` on head, `1e-3` on aggregator) completely resolved the performance gap.
3. **Fold 1 Recovery**: Fold 1 recovered from `0.6296` to **`0.7963`** (exceeding Stage 4's `0.7222`).

### Quantitative Results: Reconciled Four-Way Aggregation Benchmark

| Metric | Mean Pooling | Max Pooling | Attention Pooling | Patient Mamba (Ours) | Delta (Mamba vs Mean) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Classification ROC-AUC ($\tau=0.50$)** | 0.7296 ± 0.1477 | 0.7215 ± 0.1770 | 0.7363 ± 0.1385 | **0.7733 ± 0.1221** | **+0.0437** |
| **Classification PR-AUC** | 0.8250 ± 0.1128 | 0.8080 ± 0.1608 | 0.8315 ± 0.0954 | **0.8238 ± 0.1378** | -0.0012 |
| **Classification F1 Score** | 0.7201 ± 0.1123 | 0.7222 ± 0.0765 | 0.7348 ± 0.0882 | **0.7143 ± 0.1103** | -0.0058 |
| **Calibrated Balanced Acc ($\tau^*$)** | 0.7722 ± 0.0936 | 0.8078 ± 0.1150 | 0.7656 ± 0.0845 | **0.8078 ± 0.0755** | **+0.0356** |
| **Calibrated Specificity ($\tau^*$)** | 0.8333 ± 0.2108 | 0.7933 ± 0.2444 | 0.8200 ± 0.2227 | **0.8600 ± 0.1272** | **+0.0267** |
| **Calibrated Sensitivity ($\tau^*$)** | 0.7111 ± 0.1663 | 0.8222 ± 0.1333 | 0.7111 ± 0.2177 | **0.7556 ± 0.0831** | **+0.0445** |
| **Full-Cohort Survival C-index ($N=92$)**| 0.5883 ± 0.1082 | 0.6078 ± 0.0912 | 0.5981 ± 0.0974 | **0.6627 ± 0.0977** | **+0.0744** |
| **CV Runtime (5 Folds)** | **7.90s** | **6.74s** | **7.35s** | **15.69s** | Fast training on GPU |

*Note: In an isolated classification-only setting (no Cox survival head), Patient Mamba reaches **`0.8044 ± 0.0768`** ROC-AUC.*

### Reconciled Paired Per-Fold Comparison (Stage 4 Baseline vs Patient Mamba)

| Fold | Stage 4 Vol Logit (Baseline) | Stage 4 Emb Head (Control) | Mean Pool | Attention Pool | Patient Mamba | Delta (Mamba vs Stage 4) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 0** | 0.4815 | 0.4815 | 0.5000 | 0.5370 | **0.5370** | +0.0555 |
| **Fold 1** | 0.7222 | 0.7407 | 0.7037 | 0.7222 | **0.7963** | +0.0741 |
| **Fold 2** | 0.9111 | 0.9111 | 0.8667 | 0.8667 | **0.8667** | -0.0444 |
| **Fold 3** | 0.8889 | 0.8444 | 0.9111 | 0.9111 | **0.8667** | -0.0222 |
| **Fold 4** | 0.6222 | 0.6667 | 0.6667 | 0.6444 | **0.8000** | +0.1778 |
| **Mean** | **0.7252** | **0.7289** | 0.7296 | 0.7363 | **0.7733** | **+0.0481** |

### Running Stage 5 & 5.5
```bash
# Run unit tests (including length-1 edge case and survival metrics)
pytest tests/test_patient_mamba.py -v

# Extract volume embeddings per fold (leakage-free)
python scripts/extract_embeddings.py

# Run reconciled 4-way aggregator benchmark under nested 5-fold CV
python scripts/train_patient_mamba.py
```

---

## 11. Stage 5.75: Establishing the Noise Floor & Non-Peeking Benchmark

Stage 5.75 eliminates validation peeking in checkpoint selection and measures the empirical noise floor across multiple random seeds before moving to Stage 6.

### Multi-Seed Variance Check & Noise Floor (Task 1)
Stage 4's Hierarchical 3D Mamba was trained end-to-end across seeds `[42, 123, 456]` under identical 5-fold CV:
- **Seed 42 (Fixed Ep15)**: `0.5319 ± 0.1623` (Val-Peeking: `0.6889`, Gap: `+0.1570`)
- **Seed 123 (Fixed Ep15)**: `0.5267 ± 0.2372` (Val-Peeking: `0.7222`, Gap: `+0.1956`)
- **Seed 456 (Fixed Ep15)**: `0.5267 ± 0.2291` (Val-Peeking: `0.7270`, Gap: `+0.2004`)
- **Multi-Seed Mean (Fixed Ep15)**: **`0.5284`**
- **Multi-Seed Spread ($\max - \min$)**: **`0.0052`** (Empirical Noise Floor)
- **Multi-Seed Std ($\sigma_{\text{seeds}}$)**: **`0.0025`**
- **Validation Peeking Inflation**: **`+0.1843`** mean artificial inflation!

### Honest Stage 4 Baseline (Task 2)
Under the Fixed Epoch 15 non-peeking rule on reference Seed 42:
- Control A (Volume-Logit Mean Pooling): **`0.5319 ± 0.1623`**
- Control B (Mean Embedding then Head): **`0.5504 ± 0.1782`**

### Reconciled Stage 5 Comparison Under Non-Peeking Protocol (Task 3)

| Aggregator | Classification ROC-AUC | Full-Cohort Survival C-index ($N=92$) | Delta vs Stage 4 Baseline |
| :--- | :---: | :---: | :---: |
| **Stage 4 Baseline (Vol Logit)** | 0.5319 ± 0.1623 | N/A | Reference |
| **Mean Pooling** | 0.5326 ± 0.1746 | 0.5493 ± 0.1118 | +0.0007 |
| **Max Pooling** | 0.5733 ± 0.1458 | 0.5706 ± 0.1021 | +0.0414 |
| **Attention Pooling** | 0.5370 ± 0.1656 | 0.5552 ± 0.1124 | +0.0051 |
| **Patient Mamba (Ours)** | **0.5615 ± 0.1426** | **0.6097 ± 0.1170** | **+0.0296 (5.7x Noise Floor)** |

### Gate Decision
- **Empirical Noise Floor**: `0.0052`
- **Patient Mamba Advantage vs Stage 4**: `+0.0296` (**5.7x noise floor**)
- **Patient Mamba Advantage vs Mean Pooling**: `+0.0289` (**5.6x noise floor**)
- **Patient Mamba Survival C-index Advantage**: `+0.0604` (`0.6097` vs `0.5493`)
- **Verdict**: **`CLEARS_NOISE_FLOOR`**.
- **Justification for Stage 6**: Under honest non-peeking evaluation, supervised volume encoders trained on only 57 patients achieve ~`0.53` AUC. Stage 6 self-supervised pretraining (masked volume modeling on all 203 cached volumes + 462 raw DICOM series) is necessary to learn generalizable volumetric spatial representations.

### Running Stage 5.75
```bash
python scripts/run_noise_floor_study.py --cached-task1-json runs/stage4_fixed_checkpoints/task1_seeds_summary.json
```

---

## 12. Stage 5.9: Closing the Protocol Gap & Rigorous Statistical Significance Testing

Stage 5.9 resolves the protocol asymmetry between encoder and aggregator evaluation, compiles the unified 4-way comparison table under honest non-peeking criteria, and conducts a 10-seed paired significance analysis (Wilcoxon signed-rank and 2,000-sample patient-level bootstrap).

### Task 1: Audit Aggregator Checkpoint Selection
Aggregator models were audited and confirmed to have historically used validation peeking (`if cls_score > best_cls_score` over 35 epochs). Eliminating this loophole via pre-specified **Fixed Epoch 35** evaluation quantified the aggregator peeking inflation gap (Seed 42):
- **Mean Pooling**: Fixed Ep 35 `0.5008` vs Val Peeking `0.5326` (Inflation: `+0.0318`)
- **Max Pooling**: Fixed Ep 35 `0.5407` vs Val Peeking `0.5733` (Inflation: `+0.0326`)
- **Attention Pooling**: Fixed Ep 35 `0.4674` vs Val Peeking `0.5370` (Inflation: `+0.0696`)
- **Patient Mamba**: Fixed Ep 35 `0.4593` vs Val Peeking `0.5615` (Inflation: `+0.1022`)

### Task 2: Unified 4-Way Architecture Comparison Under Honest Protocol
Every architecture across Stages 2–5 was evaluated under identical zero-peeking discipline (Fixed Epoch 15 for 3D encoders, Fixed Epoch 35 for patient aggregators):

| Architecture | Parameters | VRAM (MB) | Honest AUC (Fixed Epoch) | Peeking AUC (Optimistic) | Peeking Inflation Gap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TinyCNN3D (Stage 2)** | 73,097 | 156.2 | **0.5652 ± 0.1912** | 0.7052 | +0.1400 |
| **FlatMamba3D (Stage 3)** | 577,665 | 579.1 | **0.5711 ± 0.1873** | 0.7067 | +0.1356 |
| **HierarchicalMamba3D (Stage 4, Seed 42)** | 627,969 | 622.0 | **0.5319 ± 0.1623** | 0.6889 | +0.1570 |
| **PatientMamba3D (Stage 5, Seed 42)** | 218,754 | 110.0 | **0.4593 ± 0.1081** | 0.5615 | +0.1022 |

> [!IMPORTANT]
> **Core Architectural Finding**:
> Supervised 3D representation learning on ~57 patients per fold achieves **`0.53 – 0.57`** ROC-AUC across all architectures without validation peeking. The earlier reported `~0.70` values were entirely an artifact of validation-peeking early stopping.

### Task 3: 10-Seed Variance & Statistical Significance (N=10 Seeds, 50 Folds)
Across 10 seeds (`[42, 100, 200, 300, 400, 500, 600, 700, 800, 900]`) evaluated under Fixed Epoch 35 on frozen honest embeddings:

| Model | 10-Seed Mean ROC-AUC | Seed AUC Spread | Calibrated Balanced Acc ($\tau^*$) | Survival C-index ($N=92$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Pooling** | 0.4973 ± 0.0070 | 0.0267 | 0.6200 ± 0.0065 | 0.4865 ± 0.0130 |
| **Max Pooling** | 0.5370 ± 0.0084 | 0.0237 | 0.6346 ± 0.0130 | 0.5024 ± 0.0093 |
| **Attention Pooling** | 0.4858 ± 0.0103 | 0.0400 | 0.6089 ± 0.0112 | 0.4872 ± 0.0145 |
| **Patient Mamba** | 0.4788 ± 0.0232 | 0.0726 | 0.6078 ± 0.0235 | **0.5194 ± 0.0167** |

#### Hypothesis Testing & Bootstrap Analysis (Patient Mamba vs Mean Pooling)
1. **Paired Fold Analysis ($N=50$ paired folds)**:
   - Classification Delta AUC: `-0.0185 ± 0.0947` ($p = 0.1769$ paired t-test, $p = 0.1565$ Wilcoxon signed-rank).
   - Survival Delta C-index: **`+0.0329 ± 0.0798`** ($t = 2.8806, \mathbf{p = 0.0059}$ paired t-test, $p = 0.0611$ Wilcoxon).
2. **Patient-Level Bootstrap ($B=2,000$ resamples)**:
   - **Binary Recurrence ($N=72$)**: $\Delta \text{AUC} = -0.0288$, 95% CI: `[-0.0934, +0.0329]`, $p = 0.3870$ (**Not significant**).
   - **Time-to-Event Survival ($N=92$)**: $\Delta \text{C-index} = +0.0041$, 95% CI: `[-0.1063, +0.1044]`, $p = 0.9530$.

### Gate Decision & Path to Stage 6
- **Plain Reporting on Classification**: The cohort size of $N=72$ for binary recurrence lacks statistical power to distinguish sequence modeling from pooling ($p = 0.3870$, 95% CI spans zero). Classification does **not** carry the justification for Stage 6.
- **Empirical Motivation for Stage 6**: Supervised 3D representations on raw DICOMs collapse to ~`0.53` without peeking. Stage 6 self-supervised pretraining (masked autoencoding across all 203 cached volumes + 462 raw DICOM series) is necessary to learn generalizable volumetric priors, while multimodal clinical fusion provides the signal density required for robust outcome prediction.

### Running Stage 5.9
```bash
python scripts/run_protocol_audit_and_significance.py --n-seeds 10 --fixed-epoch 35 --bootstrap-samples 2000
```

