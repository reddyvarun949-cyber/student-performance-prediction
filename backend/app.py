"""Flask API for the Student At-Risk Prediction project.

Run from the project root:
    python backend/app.py
The API starts on http://localhost:5000

Endpoints
    GET  /api/health    -> is the API up and is the model loaded?
    GET  /api/options   -> allowed values, ranges and the threshold (the frontend uses these to build the form)
    POST /api/predict   -> JSON with a student's week-5 details -> probability, at_risk flag, risk level, reasons

The model is trained in notebook/Student_Performance_Prediction.ipynb and loaded ONCE when the server starts.
We never retrain on a request.

Ethical use: a flag means "offer this student support", never "punish" or "label" them.
"""
import json
import math
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "student_risk_model.joblib")
INFO_PATH = os.path.join(BASE_DIR, "models", "model_info.json")

# Which website may call this API from a browser (the Vite dev server by default).
ALLOWED_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")

app = Flask(__name__)

# ---- load the model once at start-up ----
if not os.path.exists(MODEL_PATH) or not os.path.exists(INFO_PATH):
    raise SystemExit("Model files not found. Run notebook/Student_Performance_Prediction.ipynb first to create models/.")
model = joblib.load(MODEL_PATH)
with open(INFO_PATH) as f:
    info = json.load(f)

NUMERIC = info["numeric_features"]
CATEGORICAL = info["categorical_features"]
LABELS = info["feature_labels"]
THRESHOLD = info["threshold"]                 # chosen in the notebook to favour recall
HIGH_FROM = info["risk_levels"]["high_from"]


@app.after_request
def add_cors_headers(response):
    """Allow the React app (a different port = a different 'origin') to call this API."""
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def validate(payload):
    """Return (clean_row, errors). errors is a dict: field -> message a person can understand."""
    errors, row = {}, {}
    if not isinstance(payload, dict):
        return None, {"body": "Send a JSON object with the student's details."}

    for field in NUMERIC:
        low, high = info["numeric_ranges"][field]
        value = payload.get(field)
        if value is None or value == "":
            errors[field] = "This field is required."
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors[field] = "Must be a number."
            continue
        if math.isnan(number) or math.isinf(number):
            errors[field] = "Must be a real number."
        elif not (low <= number <= high):
            errors[field] = f"Must be between {low:g} and {high:g} (the range the model was trained on)."
        else:
            row[field] = number

    for field in CATEGORICAL:
        allowed = info["categories"][field]
        value = payload.get(field)
        if value is None or str(value).strip() == "":
            errors[field] = "This field is required."
        elif str(value).strip() not in allowed:
            errors[field] = f"Unknown value '{value}'. Choose one of: {', '.join(allowed)}."
        else:
            row[field] = str(value).strip()
    return row, errors


def risk_level(probability):
    """Turn a probability into a word a mentor understands."""
    if probability >= HIGH_FROM:
        return "High"
    if probability >= THRESHOLD:
        return "Medium"
    return "Low"


def reasons_for(row, max_reasons=3):
    """Simple, transparent reasons: go through the features from most to least important and
    mention those where this student is clearly on the 'risky side' of a typical (median) student.
    This is a rule of thumb, not an exact explanation of the model."""
    reasons = []
    overall = info["train_at_risk_rate"]
    for col in info["importance_order"]:
        value = row[col]
        if col in NUMERIC:
            q25, median, q75 = info["quartiles"][col]
            if info["risky_side"][col] == "low":
                noticeable, strong, word = value <= (median + q25) / 2, value <= q25, "lower"
            else:
                noticeable, strong, word = value >= (median + q75) / 2, value >= q75, "higher"
            if noticeable:
                word = ("much " if strong else "") + word
                reasons.append(f"{LABELS[col]} is {value:g}, {word} than a typical student ({round(median, 1):g}).")
        else:
            rate = info["category_risk_rate"][col].get(value, overall)
            if rate > overall * 1.2:
                reasons.append(f"{LABELS[col]}: {value}. Students in this group were at risk more often "
                               f"({rate:.0%} vs {overall:.0%} overall).")
        if len(reasons) == max_reasons:
            break
    return reasons


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": info["model_name"], "threshold": THRESHOLD})


@app.get("/api/options")
def options():
    return jsonify({
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "feature_labels": LABELS,
        "categories": info["categories"],
        "numeric_ranges": info["numeric_ranges"],
        "typical_values": {c: info["quartiles"][c][1] for c in NUMERIC},
        "threshold": THRESHOLD,
        "risk_levels": info["risk_levels"],
        "model": info["model_name"],
        "test_recall": info["test_metrics"]["recall"],
    })


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":           # browser "preflight" check before a POST
        return ("", 204)
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON (Content-Type: application/json)."}), 400

    row, errors = validate(payload)
    if errors:
        return jsonify({"error": "Some inputs are invalid.", "fields": errors}), 422

    features = pd.DataFrame([row], columns=NUMERIC + CATEGORICAL)
    probability = float(model.predict_proba(features)[0, 1])
    at_risk = probability >= THRESHOLD
    reasons = reasons_for(row)
    if at_risk:
        note = "Suggested action: a friendly mentor check-in to offer support. This is not a judgement of the student."
    else:
        note = "No alert. Teachers can still refer this student if they are worried."
    return jsonify({
        "probability": round(probability, 3),
        "at_risk": bool(at_risk),
        "risk_level": risk_level(probability),
        "threshold": THRESHOLD,
        "reasons": reasons,
        "model": info["model_name"],
        "note": note,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
