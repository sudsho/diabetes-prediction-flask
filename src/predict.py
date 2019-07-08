"""Prediction helpers used by the Flask app and CLI."""
import joblib
import numpy as np

from src.preprocess import FEATURE_COLS


def load_model(path):
    return joblib.load(path)


def predict_one(clf, features):
    """features is a dict keyed by FEATURE_COLS or a list in that order."""
    if isinstance(features, dict):
        row = [float(features[c]) for c in FEATURE_COLS]
    else:
        row = [float(x) for x in features]
    arr = np.array(row).reshape(1, -1)
    pred = int(clf.predict(arr)[0])
    if hasattr(clf, "predict_proba"):
        proba = float(clf.predict_proba(arr)[0, 1])
    else:
        proba = None
    return {"prediction": pred, "probability": proba}
