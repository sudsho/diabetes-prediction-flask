import numpy as np
from sklearn.linear_model import LogisticRegression

from src.predict import predict_one
from src.preprocess import FEATURE_COLS


def fit_dummy():
    rng = np.random.RandomState(0)
    X = rng.rand(50, len(FEATURE_COLS))
    y = (X[:, 1] > 0.5).astype(int)
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf


def test_predict_one_dict_input():
    clf = fit_dummy()
    feat = {c: 0.5 for c in FEATURE_COLS}
    out = predict_one(clf, feat)
    assert "prediction" in out
    assert out["prediction"] in (0, 1)
    assert out["probability"] is not None
    assert 0.0 <= out["probability"] <= 1.0


def test_predict_one_list_input_matches_dict():
    clf = fit_dummy()
    feat_list = [0.1, 0.9, 0.2, 0.4, 0.5, 0.3, 0.7, 0.6]
    feat_dict = dict(zip(FEATURE_COLS, feat_list))
    a = predict_one(clf, feat_list)
    b = predict_one(clf, feat_dict)
    assert a["prediction"] == b["prediction"]
