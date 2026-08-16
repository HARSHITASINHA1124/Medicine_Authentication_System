

# 1. PROJECT CONTEXT

The project is an AI + hardware + software system for preliminary
medicine authentication.

It uses:

- AS7262 spectral sensing
- controlled illumination
- Raspberry Pi
- Machine Learning
- backend software
- SQLite
- frontend dashboard
- Explainable AI

The system is NOT an ML-only project.

The Raspberry Pi is intended to host the complete deployed application:

    AS7262
       |
       v
    Raspberry Pi
       |
       +---- Backend
       |
       +---- ML inference
       |
       +---- SQLite
       |
       +---- Frontend
       |
       v
    Laptop browser

The laptop is primarily used to view the frontend and results.

---

# 2. FINAL SYSTEM GOAL

A user places a medicine sample inside a controlled sensing
compartment.

The system:

1. Illuminates the medicine consistently.
2. Captures its reflected optical response using AS7262.
3. Receives six spectral readings.
4. Calibrates the sensor data.
5. Preprocesses the readings.
6. Performs feature engineering.
7. Identifies the medicine class.
8. Compares the sample with that medicine's learned reference
   distribution.
9. Calculates an anomaly/authentication score.
10. Produces:

        VERIFIED
        SUSPICIOUS
        NOT VERIFIED

11. Generates an explanation.
12. Stores the result through the backend in SQLite.
13. Displays the result through the frontend.

---

# 3. IMPORTANT SCIENTIFIC CONSTRAINT

AS7262 provides only six spectral channels.

It does NOT directly identify the chemical composition of a medicine.

Therefore the project must NOT claim:

    "AS7262 directly detects the chemical ingredients."

The correct interpretation is:

    "The AS7262 captures an optical spectral signature and the
     ML system learns patterns associated with reference samples."

The final system is a screening/authentication prototype and is not a
replacement for laboratory chemical analysis.

---

# 4. PUBLIC DATASET

Available ASD files:

    T_Avg_ASD.xlsx
    P_Avg_ASD.xlsx
    PAM_Avg_ASD.xlsx
    K_Avg_ASD.xlsx
    NCD_Avg_ASD.xlsx
    metadata.xlsx

The ASD data contains high-resolution spectra approximately covering
350–2500 nm.

The final AS7262 hardware only provides:

    450 nm
    500 nm
    550 nm
    570 nm
    600 nm
    650 nm

Therefore an AS7262-compatible dataset was created from the ASD
spectra.

The current file is:

    medicines_as7262.csv

The wavelength representation was designed around AS7262 wavelength
bands rather than simply selecting one nearest wavelength.

---

# 5. IMPORTANT DATASET LIMITATION

The public ASD dataset is NOT a sufficiently large,
ground-truth genuine-vs-counterfeit dataset.

Therefore:

DO NOT train or describe the system as a conventional:

    spectral input -> genuine/counterfeit

supervised classifier based on the ASD data.

The intended architecture is:

    medicine identification
             +
    medicine-specific reference/anomaly detection

The ASD-derived dataset is mainly for:

- pipeline development
- testing
- feature engineering experiments
- debugging

The final deployed classifier must be trained and validated with
actual AS7262 hardware measurements.

---

# 6. TARGET MEDICINES

The project intentionally focuses on a small number of common and
easily available medicines.

Current configuration:

    paracetamol
    aspirin
    vitamin_c

The final project may use 3–4 medicines.

Do NOT redesign the system around classification of every possible
medicine.

---

# 7. CURRENT DIRECTORY

Current ML directory should look approximately like:

    ml/
    |
    +-- data/
    |   |
    |   +-- processed/
    |       |
    |       +-- medicines_as7262.csv
    |
    +-- models/
    |
    +-- src/
        |
        +-- config.py
        +-- data_loader.py
        +-- data_validation.py
        +-- calibration.py
        +-- preprocessing.py

Additional training/evaluation files will be added later.

---

# 8. FILE: config.py

Purpose:

Central configuration for:

- AS7262 feature names
- target medicines
- identifier columns
- model directory

The six AS7262 features are:

    ch450
    ch500
    ch550
    ch570
    ch600
    ch650

Current medicine classes:

    paracetamol
    aspirin
    vitamin_c

---

# 9. FILE: data_loader.py

Purpose:

Load the dataset and make the data format consistent.

The loader supports the current ASD-derived dataset and the intended
future hardware dataset.

The intended future hardware metadata fields are:

    sample_id
    medicine
    batch_id
    manufacturer
    measurement_id

and the six AS7262 channels.

The loader also handles the current dataset's older naming such as
"Sample Code".

This compatibility should be preserved unless the dataset schema is
intentionally changed.

---

# 10. FILE: data_validation.py

Current validation checks include:

- number of rows
- number of columns
- medicine class counts
- unknown medicines
- missing spectral values
- numeric conversion
- duplicate rows
- unique sample count
- spectral descriptive statistics

Future validation should add hardware-specific checks after the
sensor is operational.

Potential checks:

- saturation
- impossible values
- unstable readings
- sensor errors
- environmental measurement issues

Do not invent AS7262 physical limits without checking the actual
sensor configuration and measurement mode.

---

# 11. FILE: calibration.py

Calibration is currently a framework only.

The final hardware calibration procedure has not been established.

The intended architecture is:

    raw AS7262
          |
          v
      calibration
          |
          v
    calibrated AS7262
          |
          v
      preprocessing

Do NOT merge calibration and ML preprocessing.

The calibration procedure should be determined experimentally based
on the actual:

- sensor
- illumination
- enclosure
- measurement distance
- reference/dark procedure

If dark-reference subtraction is used, it can be implemented here.

---

# 12. FILE: preprocessing.py

This is currently the most complete ML component.

Class:

    AS7262Preprocessor

Input:

    six AS7262 channels

Output:

    engineered and standardized feature vector

Current preprocessing includes:

    1. normalized spectral intensities
    2. spectral ratios
    3. adjacent differences
    4. spectral slopes
    5. standard scaling

---

# 13. CURRENT FEATURE ENGINEERING

## Normalized intensities

Six features:

    norm_450
    norm_500
    norm_550
    norm_570
    norm_600
    norm_650

Formula:

    normalized_i = intensity_i / sum(intensities)

---

## Ratios

Six features:

    ratio_450_500
    ratio_500_550
    ratio_550_570
    ratio_570_600
    ratio_600_650
    ratio_450_650

---

## Differences

Five features:

    diff_450_500
    diff_500_550
    diff_550_570
    diff_570_600
    diff_600_650

These are calculated from normalized spectral values.

---

## Slopes

Five features:

    slope_450_500
    slope_500_550
    slope_550_570
    slope_570_600
    slope_600_650

Formula:

    slope = intensity difference / wavelength difference

This accounts for unequal wavelength spacing.

---

# 14. CURRENT TOTAL FEATURE COUNT

Current engineered representation:

    6 normalized
    + 6 ratios
    + 5 differences
    + 5 slopes

Total:

    22 features

IMPORTANT:

22 features are NOT automatically the final feature set.

The final model must compare:

    A. six raw/calibrated features

    B. six normalized features

    C. normalized + ratios

    D. all 22 features

Feature selection must be based on validation performance and
generalization.

Do not assume that more engineered features means a better model.

---

# 15. PREPROCESSING DATA LEAKAGE RULE

This rule must be preserved.

Correct:

    X_train
       |
       +--> preprocessor.fit_transform()

    X_test
       |
       +--> preprocessor.transform()

Incorrect:

    X_all
       |
       +--> preprocessor.fit_transform()
       |
       +--> train/test split

The scaler must only learn parameters from training data.

The same fitted preprocessor must later be used by the Raspberry Pi
during inference.

---

# 16. IMMEDIATE NEXT STEP

The next file to create is:

    src/eda.py

Purpose:

Explore whether the AS7262 spectral features provide useful
separation between the target medicines.

EDA should include:

- spectral profiles
- channel distributions
- feature correlations
- PCA visualization
- class balance

Do NOT use EDA results as final model performance.

---

# 17. THEN BUILD GROUPED DATA SPLITTING

Next create:

    src/split_data.py

Repeated scans from the same tablet must stay together.

Example:

    P001
       |
       +-- scan 1
       +-- scan 2
       +-- scan 3

All three must go into either:

    training

or:

    validation/test

but not both.

Otherwise the model can effectively see the same tablet during
training and testing.

---

# 18. HARDWARE DATA COLLECTION

Before final classification training, collect actual AS7262 data.

Each measurement should include:

    sample_id
    medicine
    batch_id
    manufacturer
    measurement_id
    ch450
    ch500
    ch550
    ch570
    ch600
    ch650

The collection should contain multiple tablets rather than repeatedly
scanning one tablet and treating every scan as an independent sample.

Where possible, include multiple batches and manufacturers.

This helps determine whether the model learns the medicine's spectral
characteristics rather than memorizing one particular product.

---

# 19. HARDWARE DATA IS THE CRITICAL NEXT DATA SOURCE

The ASD-derived dataset is useful for development but should not be
treated as the final training dataset.

The actual AS7262 data will determine:

- real feature distributions
- real noise
- illumination sensitivity
- sensor variability
- calibration requirements
- classifier performance
- anomaly thresholds

The final ML pipeline should therefore be designed so that the
hardware CSV can replace the ASD-derived CSV without changing the
overall architecture.

---

# 20. CLASSIFICATION MODEL

After hardware data collection and preprocessing, train a medicine
classifier.

Initial candidates:

    Logistic Regression
    SVM
    Random Forest

Input:

    processed AS7262 features

Output:

    paracetamol
    aspirin
    vitamin_c

The classifier answers:

    "Which known medicine does this sample most resemble?"

It does NOT answer:

    "Is this medicine genuine?"

---

# 21. CLASSIFIER EVALUATION

Evaluate with:

    accuracy
    precision
    recall
    F1-score
    confusion matrix

The evaluation must use grouped splits.

Where sufficient data exists, the strongest evaluation should test
generalization to unseen tablets and potentially unseen batches or
manufacturers.

Do not rely on accuracy alone.

---

# 22. ANOMALY MODEL STAGE

After classification is working, create one reference/anomaly model
per medicine.

Example:

    classifier
        |
        +--> paracetamol
        |       |
        |       +--> paracetamol reference model
        |
        +--> aspirin
        |       |
        |       +--> aspirin reference model
        |
        +--> vitamin_c
                |
                +--> vitamin C reference model

Candidate algorithms:

    Mahalanobis distance
    One-Class SVM
    Isolation Forest

The anomaly model learns the distribution of reference/genuine
samples for that specific medicine.

---

# 23. WHY MODELS ARE MEDICINE-SPECIFIC

A global anomaly model would attempt to learn one "normal medicine"
distribution across chemically and spectrally different medicines.

That is not the intended architecture.

Instead:

    Paracetamol -> Paracetamol normal distribution
    Aspirin     -> Aspirin normal distribution
    Vitamin C   -> Vitamin C normal distribution

This is more aligned with the reference-based authentication goal.

---

# 24. AUTHENTICATION LOGIC

The final inference architecture should be:

    AS7262
       |
       v
    preprocessing
       |
       v
    medicine classifier
       |
       v
    predicted medicine
       |
       v
    medicine-specific anomaly model
       |
       v
    anomaly score
       |
       v
    decision thresholds
       |
       +------------------+
       |         |        |
       v         v        v
    VERIFIED  SUSPICIOUS  NOT VERIFIED

The classifier and anomaly detector serve different purposes.

---

# 25. THRESHOLD CALIBRATION

Each medicine may have different anomaly thresholds.

Do not assume:

    one threshold for every medicine

For example:

    paracetamol_threshold
    aspirin_threshold
    vitamin_c_threshold

Thresholds must be calibrated using validation/reference data.

If genuine/counterfeit or adulterated samples become available, they
should be used to improve threshold selection and evaluate false
authentication rates.

---

# 26. COUNTERFEIT/ADULTERATED DATA LIMITATION

Currently there is no large genuine-vs-counterfeit dataset.

Therefore the system should primarily be described as:

    reference-based authentication / anomaly detection

rather than:

    supervised counterfeit classification

If known adulterated/counterfeit samples become available later,
they can be incorporated for:

- threshold selection
- validation
- supervised experiments
- false-positive/false-negative analysis

---

# 27. EXPLAINABLE AI

The final system should provide more than:

    SUSPICIOUS

It should explain why.

Possible explanation:

    Detected medicine:
    Paracetamol

    Result:
    SUSPICIOUS

    Reason:
    The spectral profile deviates from the learned
    Paracetamol reference distribution.

    Largest deviations:
    600 nm
    650 nm

    Anomaly score:
    X

    Reference threshold:
    Y

Potential XAI techniques for the classifier include:

- permutation importance
- SHAP, if appropriate for the selected model

For anomaly detection, explanation can focus on:

- channel deviations
- distance from reference mean
- largest contributing spectral channels

Do not claim chemical interpretation from individual AS7262 channels.

---

# 28. DEPLOYMENT

After training, save:

    models/
        medicine_classifier.pkl
        as7262_preprocessor.pkl
        paracetamol_reference.pkl
        aspirin_reference.pkl
        vitamin_c_reference.pkl
        thresholds.json
        model_metadata.json

The Raspberry Pi loads these files.

Training does not happen on the Raspberry Pi.

Inference does.

---

# 29. INFERENCE INTERFACE

Eventually create:

    src/inference.py

It should provide one high-level function:

    authenticate_medicine(sensor_readings)

The backend should not need to know:

- SVM implementation
- feature engineering
- scaler details
- anomaly algorithm
- threshold calculations

It should simply call the ML interface.

Expected conceptual output:

    {
        "medicine": "...",
        "result": "VERIFIED",
        "anomaly_score": ...,
        "confidence": ...,
        "explanation": ...,
        "model_version": "..."
    }

The exact schema should be aligned with the existing backend before
integration.

---

# 30. BACKEND INTEGRATION

The backend endpoint is important because the complete application
will run on Raspberry Pi.

Expected flow:

    AS7262
       |
       v
    Backend endpoint
       |
       v
    ML inference
       |
       v
    Result
       |
       +--> SQLite
       |
       +--> Frontend
       |
       v
    Laptop browser

ML should not directly write to SQLite.

The backend remains responsible for persistence.

---

# 31. RASPBERRY PI DEPLOYMENT

The Raspberry Pi will eventually run:

    frontend
    backend
    ML inference
    SQLite

The ML models should be lightweight.

This is another reason to prioritize:

- SVM
- Logistic Regression
- Random Forest
- Mahalanobis distance
- One-Class SVM
- Isolation Forest

over unnecessarily large deep-learning models.

---

# 32. COMPLETE REMAINING WORK

## Data

    [ ] Finalize hardware dataset schema
    [ ] Collect genuine/reference samples
    [ ] Collect multiple tablets
    [ ] Collect multiple batches/manufacturers where possible
    [ ] Establish repeat-scan protocol

## Hardware calibration

    [ ] Define dark/reference measurement procedure
    [ ] Measure hardware baseline
    [ ] Determine calibration transformation
    [ ] Test illumination stability
    [ ] Test measurement repeatability

## ML preprocessing

    [x] Six AS7262 channels defined
    [x] Normalization implemented
    [x] Ratios implemented
    [x] Differences implemented
    [x] Slopes implemented
    [x] Standard scaling implemented

    [ ] Test preprocessing against hardware data
    [ ] Compare feature representations
    [ ] Select final feature set

## Classification

    [ ] Grouped train/test split
    [ ] Train Logistic Regression
    [ ] Train SVM
    [ ] Train Random Forest
    [ ] Compare performance
    [ ] Select final classifier
    [ ] Save classifier
    [ ] Evaluate robustness

## Authentication

    [ ] Build paracetamol reference model
    [ ] Build aspirin reference model
    [ ] Build vitamin C reference model
    [ ] Compare anomaly algorithms
    [ ] Calibrate thresholds
    [ ] Build decision engine

## Explainability

    [ ] Classifier explanation
    [ ] Anomaly explanation
    [ ] Channel-level deviation analysis
    [ ] User-friendly explanation format

## Deployment

    [ ] Build inference.py
    [ ] Define final ML output schema
    [ ] Connect ML to backend endpoint
    [ ] Save results through SQLite
    [ ] Connect frontend
    [ ] Deploy to Raspberry Pi
    [ ] Test laptop-to-Pi access
    [ ] End-to-end scan test

---

# 33. DO NOT CHANGE THESE ARCHITECTURAL DECISIONS WITHOUT A REASON

1. AS7262 provides six physical spectral channels.

2. ASD data is a development/reference source, not equivalent to
   hardware measurements.

3. The project does not currently have sufficient genuine-vs-counterfeit
   data for a conventional supervised counterfeit classifier.

4. Medicine classification and authentication are two separate stages.

5. Each medicine gets its own reference/anomaly model.

6. Calibration is separate from ML preprocessing.

7. Preprocessing must be fitted only on training data.

8. Repeated scans of the same tablet must remain grouped during
   evaluation.

9. Training happens on a development machine.

10. Inference happens on the Raspberry Pi.

11. Backend is responsible for API handling and SQLite persistence.

12. Frontend displays the result but does not perform ML.

13. The final system is a screening/authentication prototype and
    should not claim laboratory-grade chemical verification.

---

# 34. IMMEDIATE NEXT ACTION

The project is currently at:

    Data loading
       |
       v
    Validation
       |
       v
    Calibration framework
       |
       v
    Preprocessing + feature engineering
       |
       v
    >>> CURRENT POSITION <<<

The immediate next implementation task is:

    BUILD EDA + PREPROCESSING TESTING

Then:

    GROUPED DATA SPLITTING

Then:

    CLASSIFICATION TRAINING

Do not build the anomaly models yet.

The classification pipeline should first be working correctly with
the real AS7262 hardware dataset.

Once classification is validated, proceed to the
medicine-specific reference/anomaly stage.