"""Flask app: simple HTML form -> diabetes prediction."""
import logging
import os

from flask import Flask, render_template, request

from src.predict import load_model, predict_one
from src.preprocess import FEATURE_COLS


MODEL_PATH = os.environ.get("MODEL_PATH", "models/diabetes.pkl")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

app = Flask(__name__)
_clf = None


def get_model():
    global _clf
    if _clf is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "model file not found at %s. run compare_models first." % MODEL_PATH
            )
        log.info("loading model from %s", MODEL_PATH)
        _clf = load_model(MODEL_PATH)
    return _clf


@app.route("/", methods=["GET"])
def index():
    return render_template("form.html", fields=FEATURE_COLS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {c: request.form.get(c, 0) for c in FEATURE_COLS}
        out = predict_one(get_model(), values)
    except (ValueError, RuntimeError) as exc:
        return render_template("error.html", message=str(exc)), 400
    return render_template(
        "result.html",
        values=values,
        prediction=out["prediction"],
        probability=out["probability"],
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
