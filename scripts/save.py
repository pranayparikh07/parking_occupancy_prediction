from flask import Flask, request
import csv
from datetime import datetime

app = Flask(__name__)
filename = "parking_dataset.csv"

# Create CSV headers if file does not exist
try:
    with open(filename, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["created_at", "slot_id", "status"])
except FileExistsError:
    pass

@app.route("/save", methods=["POST"])
def save():
    slot_id = request.form.get("field1")
    status  = request.form.get("field2")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, slot_id, status])

    print(f"Logged: Slot {slot_id} → {status} at {timestamp}")
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
