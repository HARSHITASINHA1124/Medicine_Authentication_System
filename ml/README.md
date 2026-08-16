# Medicine Authentication ML

This ML module is part of the Medicine Authentication System and focuses on identifying and validating pharmaceutical products using spectral sensor data, especially AS7262 measurements.

The goal is to build a machine learning pipeline that learns patterns from captured light signatures and classifies whether a medicine sample matches the expected product or should be flagged as suspicious or unknown.

## What this part does

- Loads and preprocesses spectral dataset files
- Validates input quality and consistency
- Builds calibration and preprocessing pipelines for sensor readings
- Trains and evaluates models for medicine authentication
- Exports processed datasets for downstream application use

The project is organized around these main components:

- `create_as7262_dataset.py` – generates or prepares AS7262 dataset files
- `src/data_loader.py` – loads data from local files
- `src/preprocessing.py` – handles cleaning and feature preparation
- `src/calibration.py` – sensor calibration logic
- `src/data_validation.py` – validates data integrity
- `src/config.py` – configuration for dataset paths and runtime settings

## Project structure

```text
ml/
├── create_as7262_dataset.py
├── requirements.txt
├── data/
│   ├── hardware/
│   ├── processed/
│   └── public/
├── src/
│   ├── calibration.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_validation.py
│   └── preprocessing.py
└── README.md
```

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

## Typical workflow

### 1. Prepare the dataset

```bash
python create_as7262_dataset.py
```

This script is intended to build or refresh the AS7262 training data used by the ML pipeline.

### 2. Run the preprocessing pipeline

The code under `src/` is designed to be imported and used by training scripts or notebooks. For example, you can load data with `data_loader.py`, validate it with `data_validation.py`, and normalize it with `preprocessing.py` before model training.

### 3. Train/evaluate models

Add or run your training script from the `ml` folder, using the prepared dataset and imported helpers from `src`.

## Data folders

- `data/public/` – public or externally provided data
- `data/processed/` – cleaned and transformed data ready for modeling
- `data/hardware/` – sensor/device-specific artifacts or raw captures

## Notes

- Keep raw datasets out of version control unless specifically required.
- Use `.venv` and generated artifacts locally instead of committing them.
- If you are integrating this into the main backend system, make sure the trained model files and preprocessing configuration are versioned and deployed consistently.

## Development tip

If you are extending this module, start by tracing the data flow in this order:

1. `src/config.py`
2. `src/data_loader.py`
3. `src/data_validation.py`
4. `src/preprocessing.py`
5. model training code or notebook pipeline

This keeps the pipeline easy to debug and ensures that sensor readings are validated before model training.
