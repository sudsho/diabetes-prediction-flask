"""Offline test of the Flask serving path.

Trains a tiny model into a temp path, points the app at it, and hits the
routes with the Flask test client. No network, no pre-existing pkl needed.
"""
import joblib
import numpy as np

from src.model import build_model
from src.preprocess import FEATURE_COLS


def _make_model(path):
    rng = np.random.RandomState(0)
    X = rng.rand(60, len(FEATURE_COLS))
    y = (X[:, 1] > 0.5).astype(int)
    clf = build_model("logreg", {"C": 1.0, "max_iter": 200})
    clf.fit(X, y)
    joblib.dump(clf, path)
    return clf


def _client(tmp_path, monkeypatch):
    model_path = tmp_path / "diabetes.pkl"
    _make_model(str(model_path))
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    import importlib

    import app as flask_app
    flask_app = importlib.reload(flask_app)  # re-read MODEL_PATH, reset cached model
    return flask_app.app.test_client()


def test_health(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_predict_returns_result_page(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    form = {c: "1.0" for c in FEATURE_COLS}
    r = client.post("/predict", data=form)
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Predicted:" in body


def test_index_serves_form(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    r = client.get("/")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Glucose" in body
