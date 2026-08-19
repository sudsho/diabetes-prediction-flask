"""Offline end to end smoke test for diabetes-prediction-flask.

What it does, with no network access:
1. Loads the bundled Pima CSV (data/diabetes.csv), runs the same
   zero-to-NaN + median-impute preprocessing the training code uses.
2. Trains a logistic-regression pipeline on a stratified split and prints
   the test-split metrics (accuracy / precision / recall / f1).
3. Saves the model to models/diabetes.pkl (the path the Flask app loads).
4. Boots the Flask app in-process with its test client and:
     - GET  /health   -> asserts {"status": "ok"}
     - POST /predict  -> asserts HTTP 200 and a rendered prediction page.

Run:  python scripts/smoke.py   (or: make smoke)
"""
import os
import sys

# Make the repo root importable when run as `python scripts/smoke.py`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
from sklearn.model_selection import train_test_split

from src import model as model_lib
from src import preprocess
from src.predict import predict_one
from src.preprocess import FEATURE_COLS
from src.train import evaluate

DATA_PATH = os.path.join(ROOT, "data", "diabetes.csv")
MODEL_PATH = os.path.join(ROOT, "models", "diabetes.pkl")
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# A single realistic Pima row used to exercise the serving path.
SAMPLE = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}


def train_and_save():
    print("[smoke] loading %s" % DATA_PATH)
    df = preprocess.load_csv(DATA_PATH)
    df = preprocess.replace_zeros_with_nan(df, ZERO_IS_MISSING)
    df = preprocess.median_impute(df, ZERO_IS_MISSING)
    X, y = preprocess.prepare_xy(df)
    print("[smoke] rows=%d features=%d" % (X.shape[0], X.shape[1]))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = model_lib.build_model("logreg", {"C": 1.0, "max_iter": 200})
    clf.fit(X_tr, y_tr)
    metrics = evaluate(clf, X_te, y_te)
    print(
        "[smoke] logreg test metrics: "
        "accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f"
        % (metrics["accuracy"], metrics["precision"],
           metrics["recall"], metrics["f1"])
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print("[smoke] saved model -> %s" % MODEL_PATH)
    return clf


def exercise_predict_helper(clf):
    out = predict_one(clf, SAMPLE)
    assert out["prediction"] in (0, 1), out
    assert out["probability"] is None or 0.0 <= out["probability"] <= 1.0, out
    print(
        "[smoke] predict_one -> prediction=%d probability=%s"
        % (out["prediction"], "%.3f" % out["probability"])
    )


def exercise_flask():
    # Point the app at the freshly trained model and import it.
    os.environ["MODEL_PATH"] = MODEL_PATH
    import app as flask_app

    client = flask_app.app.test_client()

    r = client.get("/health")
    assert r.status_code == 200, r.status_code
    assert r.get_json() == {"status": "ok"}, r.get_json()
    print("[smoke] GET /health -> 200 %s" % r.get_json())

    form = {k: str(v) for k, v in SAMPLE.items()}
    r = client.post("/predict", data=form)
    assert r.status_code == 200, (r.status_code, r.data[:200])
    body = r.get_data(as_text=True)
    assert "Predicted:" in body, body[:300]
    verdict = "positive" if "positive (likely diabetic)" in body else "negative"
    print("[smoke] POST /predict -> 200, verdict=%s" % verdict)


def main():
    clf = train_and_save()
    exercise_predict_helper(clf)
    exercise_flask()
    print("[smoke] OK: train + predict helper + Flask /health + /predict all passed")


if __name__ == "__main__":
    main()
