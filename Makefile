.PHONY: smoke test train serve

# Fully offline end to end check: train on the bundled CSV, print metrics,
# then exercise the Flask /health and /predict routes with the test client.
smoke:
	python scripts/smoke.py

test:
	python -m pytest -q

# Train all three models on the bundled data and save the best to models/diabetes.pkl.
train:
	python -m src.compare_models --config configs/default.yaml

# Run the Flask app (requires a trained models/diabetes.pkl; run `make smoke` or `make train` first).
serve:
	python app.py
