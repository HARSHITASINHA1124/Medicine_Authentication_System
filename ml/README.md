# Medicine Authentication ML

This ML module is part of the Medicine Authentication System and focuses on identifying and validating pharmaceutical products using spectral sensor data, especially AS7262 measurements.

The goal is to build a machine learning pipeline that learns patterns from captured light signatures and classifies medicines, then compares each sample against a reference distribution to produce an authentication decision.

## Phase 1 Architecture

The Phase 1 system uses:

1. **Multiclass Medicine Classifier** – identifies which medicine class the sample belongs to
2. **Shared Anomaly Engine** – computes similarity/anomaly between test sample and the reference profile
3. **Medicine-Specific Reference Profiles** – learned distributions of "known good" samples for each medicine

```
AS7262 Sensor
      |
      v
Calibration
      |
      v
Preprocessing & Feature Engineering
      |
      v
Multiclass Medicine Classifier
(paracetamol | aspirin | vitamin_c)
      |
      v
Predicted Medicine
      |
      v
Shared Anomaly Engine
+ Corresponding Reference Profile
      |
      v
Anomaly Score
+ Medicine-Specific Threshold
      |
      v
VERIFIED / SUSPICIOUS / NOT VERIFIED
```

### Why This Architecture?

- **Single anomaly algorithm** scales to new medicines by adding a new reference profile file
- **No need for separate anomaly models per medicine**
- **Consistent authentication methodology** across all medicines
- **Explicit reference-based design** aligned with the authentication goal

## Data Structure and Sample Grouping

Understanding the metadata is critical for preventing data leakage during training.

### Key Fields

- **sample_id** – The individual physical tablet being tested (e.g., "P001")
- **measurement_id** – Each AS7262 scan/reading from that sample (e.g., "1", "2", "3")
- **batch_id** – Manufacturing batch identifier (e.g., "B20230315")
- **manufacturer** – Company name (metadata only, not an ML feature)
- **medicine** – Target class (paracetamol, aspirin, vitamin_c)
- **ch450, ch500, ch550, ch570, ch600, ch650** – AS7262 spectral channels

### Critical Rule: Grouped Splits

```
CORRECT (no data leakage):
  Tablet P001 (all measurement_ids) → TRAINING SET
  Tablet P002 (all measurement_ids) → TRAINING SET
  Tablet P003 (all measurement_ids) → VALIDATION SET
  Tablet P004 (all measurement_ids) → VALIDATION SET

INCORRECT (data leakage):
  Tablet P001, measurement_1 → TRAINING
  Tablet P001, measurement_2 → VALIDATION
  ❌ Same physical tablet in both sets!
```

All measurements from the same sample_id must stay together.

## Current Dataset

- **medicines_as7262.csv** – derived from public ASD spectral data, scaled to AS7262 wavelengths
- **Purpose** – development, testing, feature engineering experiments
- **Limitation** – NOT a ground-truth genuine-vs-counterfeit dataset
- **Final training** – will use actual AS7262 hardware measurements

The ASD-derived dataset is useful for pipeline development but should not be treated as the final training dataset. The actual AS7262 data will determine real feature distributions, noise characteristics, and classifier performance.

## What this ML part does

- Loads and preprocesses spectral dataset files
- Validates input quality and consistency
- Builds calibration and preprocessing pipelines for sensor readings
- Trains and evaluates multiclass medicine classifiers
- Extracts medicine-specific reference profiles
- Builds shared anomaly/authentication engine
- Produces authentication decisions with explanations
- Supports serialization for deployment to Raspberry Pi

The project is organized around these main components:

- `create_as7262_dataset.py` – generates or prepares AS7262 dataset files
- `src/config.py` – central configuration (medicine classes, feature names)
- `src/data_loader.py` – loads data from CSV with format compatibility
- `src/data_validation.py` – validates dataset integrity and schema
- `src/calibration.py` – sensor calibration framework (currently a skeleton)
- `src/preprocessing.py` – feature engineering (normalization, ratios, slopes, etc.)
- `src/eda.py` – exploratory data analysis and feature separation testing

## Project structure

```text
ml/
├── README.md
├── .gitignore
├── requirements.txt
├── create_as7262_dataset.py
│
├── data/
│   ├── hardware/
│   ├── processed/
│   │   └── medicines_as7262.csv
│   └── public/
│
├── models/
│   └── (trained models deployed here after training)
│
├── eda_results/
│   └── (exploratory data analysis outputs)
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── calibration.py
    ├── data_loader.py
    ├── data_validation.py
    ├── preprocessing.py
    ├── eda.py
    ├── split_data.py (future: grouped data splitting)
    └── inference.py (future: high-level API for backend integration)
```

## Implementation Status

### ✅ Completed

- Data loading and compatibility layer
- Input validation framework
- Feature engineering (preprocessing.py with 22 engineered features)
- EDA and feature separation analysis
- Configuration management
- Dataset preparation script

### 🔄 In Progress

- Classification model training (next: implement src/split_data.py)
- Grouped train/test splitting

### 📋 Planned

- Multiclass medicine classifier training
- Reference profile extraction per medicine
- Shared anomaly engine implementation
- Threshold calibration
- Inference interface (src/inference.py)
- Integration with backend

## Prerequisites

- Python 3.10+
- pip
- A local virtual environment is recommended

## Quick start

From the project root:

```bash
cd ml
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or .venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## Typical Development Workflow

### 1. Prepare the dataset

```bash
python create_as7262_dataset.py
```

Generates or refreshes the AS7262-compatible dataset.

### 2. Validate data quality

```python
from src.data_loader import load_data
from src.data_validation import validate_dataset

df = load_data("data/processed/medicines_as7262.csv")
validate_dataset(df)
```

### 3. Explore feature separation (EDA)

```bash
# Run exploratory data analysis
python -c "from src.eda import *; run_all_eda()"
```

Outputs in `eda_results/` include channel distributions, correlations, and feature rankings.

### 4. Split data properly (prevent leakage)

```python
# Future: src/split_data.py will handle grouped splitting
# by sample_id and optionally by batch_id
from src.split_data import grouped_train_test_split

train_idx, test_idx = grouped_train_test_split(
    df, 
    group_key="sample_id",
    test_size=0.2
)
```

### 5. Train classifier and extract reference profiles

```python
# Future: Model training script
# Should:
# 1. Split by sample_id (no data leakage)
# 2. Preprocess only on training data
# 3. Train multiclass classifier
# 4. Extract reference profile for each medicine
# 5. Save classifier + preprocessor + profiles + thresholds
```

### 6. Evaluate with proper grouping

Evaluation must respect sample_id grouping to test real generalization.

## Important Scientific Notes

### The AS7262 Hardware

The system uses the AS7262 sensor which provides **only six spectral channels**:

- 450 nm (violet)
- 500 nm (green)
- 550 nm (yellow)
- 570 nm (yellow-orange)
- 600 nm (orange)
- 650 nm (red)

This is NOT sufficient for chemical composition identification. The system classifies based on learned **optical spectral patterns**, not chemistry.

### System Limitations and Correct Framing

**This system is NOT:**

- A chemical analyzer that identifies molecular composition
- A guarantee of genuine vs. counterfeit medicine
- Laboratory-grade verification
- A replacement for professional pharmaceutical analysis

**This system IS:**

- A screening/authentication prototype
- Reference-based optical signature matching
- Pattern recognition trained on reference samples
- A preliminary assessment tool

### Dataset Limitations

The current ASD-derived dataset:

- Is derived from public spectral data, NOT genuine vs. counterfeit samples
- Is NOT suited for training a supervised "counterfeit classifier"
- Is useful for development, testing, feature engineering, and debugging
- Should NOT be treated as the final training dataset

**Final validation must use actual AS7262 hardware measurements.**

### Data Leakage Prevention

Never allow the same physical tablet (sample_id) to appear in both training and test sets, even if the measurements (measurement_ids) differ.

## Phase 1 vs. Future Phases

### Phase 1 (Current)

- 3–4 target medicines
- Multiclass classifier
- Shared anomaly engine with reference profiles
- No counterfeit/adulterated data
- Deployment to Raspberry Pi

### Future Phases

- More medicines (reuse same anomaly engine)
- Counterfeit/adulterated sample collection and analysis
- Hardware-specific calibration refinement
- Batch-aware or manufacturer-aware models
- User interface / mobile app
- Explainability enhancements (SHAP, feature importance)

## Data Directories

### data/public/

Public or externally provided reference data (e.g., ASD spectra).

### data/processed/

Cleaned and transformed data ready for modeling.

Current file: `medicines_as7262.csv` (ASD-derived, development only)

### data/hardware/

Actual AS7262 hardware measurements (to be populated during Phase 1).

Format: CSV with fields sample_id, medicine, batch_id, manufacturer, measurement_id, ch450, ch500, ch550, ch570, ch600, ch650

### models/

Trained models, preprocessors, reference profiles, and thresholds.

Phase 1 deployment will include:

- `medicine_classifier.pkl` – multiclass classifier
- `as7262_preprocessor.pkl` – fitted feature engineering + scaling
- `anomaly_engine.pkl` – shared anomaly algorithm
- `paracetamol_reference.pkl` – reference distribution
- `aspirin_reference.pkl` – reference distribution
- `vitamin_c_reference.pkl` – reference distribution
- `thresholds.json` – medicine-specific authentication thresholds
- `model_metadata.json` – version, training date, feature info

## Development Guidelines

### Data Leakage Prevention Rule

**Never** fit preprocessor/scaler on all data then split:

```python
# ❌ WRONG
X_all.fit_transform()  # learns from ALL data including test
train_test_split()

# ✅ CORRECT
X_train, X_test = train_test_split(X_all, groups=sample_id)
X_train_processed = preprocessor.fit_transform(X_train)  # learns ONLY from train
X_test_processed = preprocessor.transform(X_test)         # uses train parameters
```

### Configuration and Constants

All configurable values live in `src/config.py`:

- Target medicines
- AS7262 feature names
- File paths
- Model directory

### Feature Engineering Pipeline

Trace the code flow in this order:

1. `src/config.py` – constants and configuration
2. `src/data_loader.py` – load and normalize column names
3. `src/data_validation.py` – check data quality
4. `src/calibration.py` – apply hardware calibration (framework only)
5. `src/preprocessing.py` – feature engineering and scaling
6. `src/eda.py` – exploratory analysis
7. Training scripts – classify, extract reference profiles, build anomaly engine

## Notes

- Keep raw datasets out of version control unless required
- Use `.venv` and generated artifacts locally (see .gitignore)
- Model files and preprocessing config must be versioned and deployed together
- Ensure grouped splitting is used in all train/test/validation operations
- The backend is responsible for API integration; ML only provides inference
- Raspberry Pi deployment should not retrain; only run inference
