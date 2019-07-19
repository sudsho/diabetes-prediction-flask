"""Print a sklearn classification report and confusion matrix for a saved model."""
import argparse

import joblib
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src import preprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", default="models/diabetes.pkl")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = preprocess.load_csv(cfg["data"]["path"])
    df = preprocess.replace_zeros_with_nan(df, cfg["data"]["zero_is_missing"])
    df = preprocess.median_impute(df, cfg["data"]["zero_is_missing"])
    X, y = preprocess.prepare_xy(df)

    sp = cfg["split"]
    _, X_te, _, y_te = train_test_split(
        X, y,
        test_size=sp["test_size"],
        random_state=sp["random_state"],
        stratify=y if sp.get("stratify", True) else None,
    )

    clf = joblib.load(args.model)
    pred = clf.predict(X_te)

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, pred))
    print()
    print(classification_report(y_te, pred, target_names=["no_diabetes", "diabetes"]))


if __name__ == "__main__":
    main()
