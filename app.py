"""Flask app: simple HTML form -> diabetes prediction."""
import os

from flask import Flask, render_template, request

from src.predict import load_model, predict_one
from src.preprocess import FEATURE_COLS


MODEL_PATH = os.environ.get("MODEL_PATH", "models/diabetes.pkl")

app = Flask(__name__)
_clf = None


def get_model():
    global _clf
    if _clf is None:
        _clf = load_model(MODEL_PATH)
    return _clf


@app.route("/", methods=["GET"])
def index():
    return render_template("form.html", fields=FEATURE_COLS)


@app.route("/predict", methods=["POST"])
def predict():
    values = {c: request.form.get(c, 0) for c in FEATURE_COLS}
    out = predict_one(get_model(), values)
    return render_template(
        "result.html",
        values=values,
        prediction=out["prediction"],
        probability=out["probability"],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
