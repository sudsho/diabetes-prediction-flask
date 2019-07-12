"""Train a single model based on the YAML config and save the joblib pickle."""
import argparse
import json
import logging
import os

import joblib
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src import model as model_lib
from src import preprocess


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="logreg",
                   help="logreg, random_forest, or xgboost")
    return p.parse_args()


def evaluate(clf, X_test, y_test):
    pred = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log.info("loading %s", cfg["data"]["path"])
    df = preprocess.load_csv(cfg["data"]["path"])
    df = preprocess.replace_zeros_with_nan(df, cfg["data"]["zero_is_missing"])
    df = preprocess.median_impute(df, cfg["data"]["zero_is_missing"])
    X, y = preprocess.prepare_xy(df)
    log.info("rows=%d features=%d", X.shape[0], X.shape[1])

    sp = cfg["split"]
    stratify = y if sp.get("stratify", True) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=sp["test_size"],
        random_state=sp["random_state"],
        stratify=stratify,
    )
    log.info("train=%d test=%d", len(X_tr), len(X_te))

    params = cfg["models"][args.model]
    clf = model_lib.build_model(args.model, params)
    log.info("fitting %s", args.model)
    clf.fit(X_tr, y_tr)

    metrics = evaluate(clf, X_te, y_te)
    log.info("metrics: %s", metrics)

    out_dir = os.path.dirname(cfg["output"]["model_path"]) or "."
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(clf, cfg["output"]["model_path"])
    with open(cfg["output"]["metrics_path"], "w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("saved %s", cfg["output"]["model_path"])


if __name__ == "__main__":
    main()
