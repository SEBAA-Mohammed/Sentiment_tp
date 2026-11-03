"""
app.py - Step 5: Flask backend (API + page).

- Loads the saved vectorizer + 3 models once at startup.
- POST /predict  -> {"text": "...", "model": "svm|naive_bayes|random_forest"}
- GET  /         -> serves docs/index.html (handy for local testing)
- GET  /health   -> "ok" (used to wake up / check the server)

Local run:   python app.py        -> http://127.0.0.1:5000
On Render:   gunicorn app:app --workers 1 --timeout 120
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from preprocessing import clean_text

MODELS_DIR = Path("models")
MODEL_FILES = {
    "svm": "svm.joblib",
    "naive_bayes": "naive_bayes.joblib",
    "random_forest": "random_forest.joblib",
}

# The HTML page lives in docs/ because GitHub Pages serves that folder.
app = Flask(__name__, template_folder="docs")

# CORS lets the page hosted on GitHub Pages call this API on Render.
# "*" = any website may call it (fine for a school project).
CORS(app, resources={r"/predict": {"origins": os.environ.get("ALLOWED_ORIGIN", "*")}})

# Load everything once at startup (not on every request)
try:
    vectorizer = joblib.load(MODELS_DIR / "vectorizer.joblib")
    models = {key: joblib.load(MODELS_DIR / f) for key, f in MODEL_FILES.items()}
except FileNotFoundError:
    sys.exit("Model files not found in 'models/'. Run `python train.py` first.")


def class_probabilities(model, X) -> dict:
    """Return {class: score}. SVM has no probabilities, so we apply a softmax
    to its decision scores (an approximate confidence, not a true probability)."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[0]
    else:
        d = model.decision_function(X)[0]
        e = np.exp(d - d.max())
        scores = e / e.sum()
    return {cls: round(float(s), 4) for cls, s in zip(model.classes_, scores)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return "ok"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    model_key = data.get("model", "svm")

    if model_key not in models:
        return jsonify(error=f"Unknown model '{model_key}'. Choose: {list(models)}"), 400
    if not text:
        return jsonify(error="Please enter some text."), 400

    cleaned = clean_text(text)
    if not cleaned:
        return jsonify(error="No usable words found in the text."), 400

    X = vectorizer.transform([cleaned])
    model = models[model_key]
    sentiment = str(model.predict(X)[0])
    probs = class_probabilities(model, X)

    return jsonify(sentiment=sentiment, model=model_key,
                   confidence=probs[sentiment], probabilities=probs)


if __name__ == "__main__":
    app.run(debug=True)
