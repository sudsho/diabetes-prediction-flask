"""Train all three models on the same split, print metrics, save the winner."""
import argparse
import json
import logging
import os

import joblib
import yaml
from sklearn.model_selection import train_test_split

from src import model as model_lib
from src import preprocess
from src.train import evaluate


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compare")


MODELS_TO_TRY = ["logreg", "random_forest", "xgboost"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = preprocess.load_csv(cfg["data"]["path"])
    df = preprocess.replace_zeros_with_nan(df, cfg["data"]["zero_is_missing"])
    df = preprocess.median_impute(df, cfg["data"]["zero_is_missing"])
    X, y = preprocess.prepare_xy(df)

    sp = cfg["split"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=sp["test_size"],
        random_state=sp["random_state"],
        stratify=y if sp.get("stratify", True) else None,
    )

    results = {}
    best = None
    for name in MODELS_TO_TRY:
        params = cfg["models"][name]
        clf = model_lib.build_model(name, params)
        log.info("fitting %s", name)
        clf.fit(X_tr, y_tr)
        m = evaluate(clf, X_te, y_te)
        results[name] = m
        log.info("%s -> %s", name, m)
        if best is None or m["f1"] > results[best[0]]["f1"]:
            best = (name, clf)

    log.info("winner: %s", best[0])
    out_dir = os.path.dirname(cfg["output"]["model_path"]) or "."
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(best[1], cfg["output"]["model_path"])
    with open(cfg["output"]["metrics_path"], "w") as fh:
        json.dump({"all": results, "winner": best[0]}, fh, indent=2)


if __name__ == "__main__":
    main()
