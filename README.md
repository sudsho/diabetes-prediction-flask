# diabetes-prediction-flask

Predicting diabetes onset on the Pima Indians Diabetes dataset and serving the
trained model behind a Flask form.

## Problem

Given eight diagnostic measurements, predict whether a patient has diabetes
(binary classification). Dataset: well-known Pima Indians Diabetes Database
from the UCI ML Repository / Kaggle.

Source: https://www.kaggle.com/uciml/pima-indians-diabetes-database

## Setup

```
python3.7 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train all three models and pick the best:
```
python -m src.compare_models --config configs/default.yaml
```

Run server:
```
python app.py
```

Open http://localhost:5000 and fill in the eight medical fields.

## Approach

1. Load the CSV (`data/diabetes.csv`, 768 rows, 8 features + 1 label).
2. Pima dataset has zeros in `Glucose`, `BloodPressure`, `SkinThickness`,
   `Insulin`, `BMI` that are really missing values. Replace with NaN, then
   impute the column median.
3. Train/test split, stratified on `Outcome`.
4. Compare three models: logistic regression (with StandardScaler),
   random forest, XGBoost.
5. Pick the best by F1 on the test split, save the joblib pickle to
   `models/diabetes.pkl`.
6. Wrap the saved model in a small Flask app (one form, one result page).
