from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load trained model
pipeline, slot_names = joblib.load("parking_prob_model.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    expected_free = None
    if request.method == "POST":
        try:
            # Accept fractional hours
            hour = float(request.form["hour"])  
            day = int(request.form["day"])

            # Prepare input
            X_new = pd.DataFrame([[hour, day]], columns=["hour", "dayofweek"])
            proba = pipeline.predict(X_new)[0]

            # Map slot → free probability
            slot_free_prob = {name: p * 100 for name, p in zip(slot_names, proba)}
            sorted_slots = dict(sorted(slot_free_prob.items(), key=lambda x: x[1], reverse=True))

            # Expected available slots
            expected_free = sum(proba)

            # Top 10 results
            predictions = list(sorted_slots.items())[:10]

        except Exception as e:
            predictions = [("Error", str(e))]

    return render_template("index.html", predictions=predictions, expected_free=expected_free)

if __name__ == "__main__":
    app.run(debug=True)
