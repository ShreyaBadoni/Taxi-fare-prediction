import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load model bundle (model + feature list saved together) ──
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model      = bundle["model"]
    features   = bundle["features"]
    model_name = bundle.get("model_name", "Unknown")
    metrics    = bundle.get("metrics", {})
    print(f"✓ Loaded: {model_name}  |  R²={metrics.get('R2', 'N/A'):.4f}")
except FileNotFoundError:
    raise RuntimeError(
        f"Model file '{MODEL_PATH}' not found. "
        "Run train_model.py first to generate it."
    )

# ── Validation helpers ────────────────────────────────────────
def validate_inputs(miles, seconds):
    errors = []
    if miles <= 0:
        errors.append("Trip distance must be greater than 0 miles.")
    if miles > 100:
        errors.append("Trip distance seems too large (>100 miles). Please double-check.")
    if seconds <= 0:
        errors.append("Trip duration must be greater than 0 seconds.")
    if seconds > 18000:  # 5 hours
        errors.append("Trip duration seems too large (>5 hours). Please double-check.")
    return errors

def build_feature_row(miles, seconds):
    speed = miles / (seconds / 3600 + 1e-6)
    row = {
        "TRIP_MILES":    miles,
        "TRIP_SECONDS":  seconds,
        "TRIP_SPEED":    speed,
        "TRIP_MINUTES":  seconds / 60,
        "IS_SHORT_TRIP": 1 if miles < 2 else 0,
        # Time features default to 0 if not provided by user
        "PICKUP_HOUR":   0,
        "DAY_OF_WEEK":   0,
        "IS_RUSH_HOUR":  0,
        "IS_WEEKEND":    0,
    }
    # Return only the features the model was trained on
    return {k: row[k] for k in features if k in row}

# ── Routes ────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    errors = []
    form_data = {}

    if request.method == "POST":
        try:
            miles   = float(request.form.get("miles", 0))
            seconds = float(request.form.get("seconds", 0))
            hour    = int(request.form.get("hour", 0))
            weekend = int(request.form.get("weekend", 0))

            form_data = {"miles": miles, "seconds": seconds, "hour": hour, "weekend": weekend}
            errors = validate_inputs(miles, seconds)

            if not errors:
                row = build_feature_row(miles, seconds)

                # Apply time features if model supports them
                if "PICKUP_HOUR" in features:
                    row["PICKUP_HOUR"]  = hour
                    row["IS_RUSH_HOUR"] = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
                if "IS_WEEKEND" in features:
                    row["IS_WEEKEND"] = weekend

                df_input = pd.DataFrame([row])[features]
                predicted_fare = model.predict(df_input)[0]
                predicted_fare = max(2.25, round(predicted_fare, 2))  # Min fare = $2.25

                speed_mph = miles / (seconds / 3600 + 1e-6)
                result = {
                    "fare":     predicted_fare,
                    "miles":    miles,
                    "minutes":  round(seconds / 60, 1),
                    "speed":    round(speed_mph, 1),
                    "model":    model_name,
                    "r2":       round(metrics.get("R2", 0), 4),
                    "rmse":     round(metrics.get("RMSE", 0), 2),
                }

        except ValueError:
            errors.append("Invalid input. Please enter numeric values only.")
        except Exception as e:
            errors.append(f"An unexpected error occurred: {str(e)}")

    return render_template(
        "index.html",
        result=result,
        errors=errors,
        form_data=form_data,
        model_name=model_name,
        metrics=metrics,
        features=features,
    )


# ── API endpoint (bonus: for future mobile/JS use) ────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data    = request.get_json(force=True)
        miles   = float(data.get("miles", 0))
        seconds = float(data.get("seconds", 0))

        errors = validate_inputs(miles, seconds)
        if errors:
            return jsonify({"error": errors}), 400

        row = build_feature_row(miles, seconds)
        df_input = pd.DataFrame([row])[features]
        fare = max(2.25, round(float(model.predict(df_input)[0]), 2))

        return jsonify({
            "predicted_fare_usd": fare,
            "model_used": model_name,
            "r2_score": round(metrics.get("R2", 0), 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
