"""Prediction helpers used by the Flask app and CLI."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

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


def predict_batch(clf, df):
    """df has FEATURE_COLS, returns the same df with prediction column added."""
    X = df[FEATURE_COLS].values
    df = df.copy()
    df["prediction"] = clf.predict(X)
    if hasattr(clf, "predict_proba"):
        df["probability"] = clf.predict_proba(X)[:, 1]
    return df


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/diabetes.pkl")
    p.add_argument("--csv", help="csv to predict on (must have FEATURE_COLS)")
    p.add_argument("--out", help="where to save predictions csv")
    args = p.parse_args()

    clf = load_model(args.model)
    if args.csv:
        df = pd.read_csv(args.csv)
        out = predict_batch(clf, df)
        if args.out:
            out.to_csv(args.out, index=False)
        else:
            print(out.to_csv(index=False))
    else:
        # interactive: read 8 floats from stdin and print json
        vals = input("enter 8 features comma-separated: ").split(",")
        print(json.dumps(predict_one(clf, vals)))


if __name__ == "__main__":
    _cli()
