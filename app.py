from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

# Load trained model
pipeline, slot_names = joblib.load("parking_prob_model.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    # Landing page no longer renders predictions
    return render_template("index.html")

def compute_predictions(hour: float, day: int, top_n: int = 10):
    X_new = pd.DataFrame([[hour, day]], columns=["hour", "dayofweek"])
    proba = pipeline.predict(X_new)[0]
    slot_free_prob = {name: p * 100 for name, p in zip(slot_names, proba)}
    sorted_slots = dict(sorted(slot_free_prob.items(), key=lambda x: x[1], reverse=True))
    expected_free = float(sum(proba))
    predictions = list(sorted_slots.items())[:top_n]
    return predictions, expected_free

@app.route("/api/predictions", methods=["GET"])
def api_predictions():
    try:
        hour = float(request.args.get("hour"))
        day = int(request.args.get("day"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing hour/day parameters"}), 400
    try:
        predictions, expected_free = compute_predictions(hour, day)
        return jsonify({
            "hour": hour,
            "day": day,
            "expected_free": expected_free,
            "predictions": [ {"slot": s, "prob": p} for s,p in predictions ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predictions", methods=["GET"])
def predictions_page():
    # Page will fetch data via JS using query params
    hour = request.args.get("hour")
    day = request.args.get("day")
    return render_template("predictions.html", hour=hour, day=day)

if __name__ == "__main__":
    app.run(debug=True)
