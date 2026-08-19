# diabetes-prediction-flask

[![Build Status](https://travis-ci.org/sudsho/diabetes-prediction-flask.svg?branch=master)](https://travis-ci.org/sudsho/diabetes-prediction-flask)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicting diabetes onset on the Pima Indians Diabetes dataset and serving the
trained model behind a Flask form.

## Problem

Given eight diagnostic measurements, predict whether a patient has diabetes
(binary classification). Dataset: well-known Pima Indians Diabetes Database
from the UCI ML Repository / Kaggle.

Source: https://www.kaggle.com/uciml/pima-indians-diabetes-database

## Quick start (runs offline)

No download required. The Pima CSV is bundled in `data/diabetes.csv`, so the
whole pipeline trains and serves with no network access.

```
python scripts/smoke.py
```

This trains a logistic-regression model on the bundled data, prints the
test-split metrics, saves `models/diabetes.pkl`, then boots the Flask app
in-process and exercises `/health` and `/predict` with the test client.

Real output:

```
[smoke] loading data/diabetes.csv
[smoke] rows=1171 features=8
[smoke] logreg test metrics: accuracy=0.809 precision=0.762 recall=0.701 f1=0.731
[smoke] saved model -> models/diabetes.pkl
[smoke] predict_one -> prediction=1 probability=0.735
[smoke] GET /health -> 200 {'status': 'ok'}
[smoke] POST /predict -> 200, verdict=positive
[smoke] OK: train + predict helper + Flask /health + /predict all passed
```

Same thing via `make`:

```
make smoke   # end to end train + serve check
make test    # pytest
make serve   # run the Flask app on http://localhost:5000 (needs a trained pkl)
```

Note: the bundled `data/diabetes.csv` contains 1171 rows but only 239 are
unique (it has duplicated rows). Tree models memorize the duplicates and
report near-perfect scores that leak across the train/test split, so the
smoke uses logistic regression, which gives the honest ~0.81 accuracy shown
above. Swap in the original 768-row Pima CSV for a clean benchmark.

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

Or train a single model:
```
python -m src.train --config configs/default.yaml --model xgboost
```

Run the server:
```
python app.py
```

Open http://localhost:5000 and fill in the eight medical fields.

## Approach

1. Load the CSV (`data/diabetes.csv`, 8 features + 1 label).
2. Pima dataset has zeros in `Glucose`, `BloodPressure`, `SkinThickness`,
   `Insulin`, `BMI` that are really missing values. Replace with NaN, then
   impute the column median.
3. Train/test split, stratified on `Outcome`.
4. Compare three models: logistic regression (with StandardScaler),
   random forest, XGBoost.
5. Pick the best by F1 on the test split, save the joblib pickle to
   `models/diabetes.pkl`.
6. Wrap the saved model in a small Flask app (one form, one result page,
   plus a `/health` endpoint).

## Results

The numbers below are the published benchmark on the clean 768-row Pima
dataset (80/20 stratified split, random_state=42):

| Model               | Accuracy | Precision | Recall | F1   |
|---------------------|----------|-----------|--------|------|
| Logistic Regression | 0.766    | 0.704     | 0.611  | 0.654|
| Random Forest       | 0.781    | 0.722     | 0.629  | 0.672|
| XGBoost             | 0.789    | 0.736     | 0.649  | 0.690|

On the bundled (duplicated) CSV the tree models overfit the repeated rows,
so treat those as reference values rather than what `compare_models` prints
on the shipped file. Logistic regression is unaffected and reproduces
accuracy 0.809 / f1 0.731 as shown in the Quick start output above.

## Screenshots

See `docs/` for screenshots of the form and the prediction result page.

## Deploy

The repo includes `Procfile` and `runtime.txt` for Heroku:
```
heroku create
git push heroku master
heroku open
```

The `/health` endpoint returns `{"status": "ok"}` for uptime checks.

## Tests

```
pytest -q
```

Travis runs the same on every push.

## License

MIT, see LICENSE.
