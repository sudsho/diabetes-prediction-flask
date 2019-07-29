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

1. Load the CSV (`data/diabetes.csv`, 768 rows, 8 features + 1 label).
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

80/20 stratified split, random_state=42:

| Model               | Accuracy | Precision | Recall | F1   |
|---------------------|----------|-----------|--------|------|
| Logistic Regression | 0.766    | 0.704     | 0.611  | 0.654|
| Random Forest       | 0.781    | 0.722     | 0.629  | 0.672|
| XGBoost             | 0.789    | 0.736     | 0.649  | 0.690|

XGBoost wins on F1 and is the default saved model.

## Screenshots

See `docs/` (placeholder, screenshots will be added once deployed).

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
