from flask import Flask, request
from datetime import datetime
import csv

app = Flask(__name__)
filename = "parking_dataset.csv"

# Initialize CSV with headers if not exists
try:
    with open(filename, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["created_at", "field1", "field2"])
except:
    pass

@app.route("/update", methods=["POST"])
def update():
    event = request.form.get("event")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Default values
    slot_id = "-"
    availability = "-"

    # Parse event string: SLOT,1,OCCUPIED
    if event.startswith("SLOT"):
        parts = event.split(",")
        slot_id = parts[1]
        availability = parts[2]

    # Save to dataset
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, slot_id, availability])

    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)