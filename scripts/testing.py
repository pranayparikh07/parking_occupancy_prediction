import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import sys
import itertools
import threading
from sklearn.metrics import (
    f1_score, mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, explained_variance_score
)
from sklearn.model_selection import cross_val_score, KFold
import base64
from io import BytesIO

# ----------------------------
# Simple loading spinner for UX
# ----------------------------
class Spinner:
    def __init__(self, message="Processing"):
        self.message = message
        self.spinner = itertools.cycle(['|', '/', '-', '\\'])
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()

    def _spin(self):
        while self.running:
            sys.stdout.write(f"\r{self.message}... {next(self.spinner)}")
            sys.stdout.flush()
            time.sleep(0.1)

    def stop(self, success_message="✅ Done"):
        self.running = False
        time.sleep(0.2)
        sys.stdout.write(f"\r{success_message}\n")
        sys.stdout.flush()

# ----------------------------
# Step 1: Load Data
# ----------------------------
print("🚀 Starting Model Evaluation Report Generation...\n")
spinner = Spinner("Loading and preparing dataset")
spinner.start()
df = pd.read_csv("SPSIRDATA.csv")
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["hour"] = df["created_at"].dt.hour
df["dayofweek"] = df["created_at"].dt.dayofweek
time.sleep(1.5)
spinner.stop("📂 Dataset loaded successfully")

spinner = Spinner("Aggregating and transforming data")
spinner.start()
prob_table = (
    df.groupby(["hour", "dayofweek", "field1"])["field2"]
      .mean()
      .reset_index()
)
pivot_df = prob_table.pivot_table(
    index=["hour", "dayofweek"],
    columns="field1",
    values="field2"
).fillna(0).reset_index()
time.sleep(1.2)
spinner.stop("🔄 Data prepared and pivoted for training")

# ----------------------------
# Step 2: Load Trained Model
# ----------------------------
spinner = Spinner("Loading trained model")
spinner.start()
pipeline, slot_names = joblib.load("parking_prob_model.pkl")
time.sleep(1.2)
spinner.stop("🧠 Model loaded successfully")

X = pivot_df[["hour", "dayofweek"]]
y = pivot_df.drop(columns=["hour", "dayofweek"])

# ----------------------------
# Step 3: Predictions and Evaluation
# ----------------------------
spinner = Spinner("Generating predictions and evaluating model")
spinner.start()
y_pred = pipeline.predict(X)
y_true_bin = (y >= 0.5).astype(int)
y_pred_bin = (y_pred >= 0.5).astype(int)

f1_scores = {slot: f1_score(y_true_bin[slot], y_pred_bin[:, i]) for i, slot in enumerate(y.columns)}
precision_scores = {slot: precision_score(y_true_bin[slot], y_pred_bin[:, i]) for i, slot in enumerate(y.columns)}
recall_scores = {slot: recall_score(y_true_bin[slot], y_pred_bin[:, i]) for i, slot in enumerate(y.columns)}

avg_f1 = np.mean(list(f1_scores.values()))
avg_precision = np.mean(list(precision_scores.values()))
avg_recall = np.mean(list(recall_scores.values()))
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
evs = explained_variance_score(y, y_pred)
accuracy = accuracy_score(y_true_bin.values.flatten(), y_pred_bin.flatten())

kf = KFold(n_splits=5, shuffle=True, random_state=42)
try:
    cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
    bias = 1 - np.mean(cv_scores)
    variance = np.std(cv_scores)
except Exception:
    bias, variance = None, None

time.sleep(2)
spinner.stop("📈 Model evaluation metrics computed")

# ----------------------------
# Step 4: Generating Graphs
# ----------------------------
spinner = Spinner("Generating visual charts")
spinner.start()

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# Scatter plot
fig1, ax1 = plt.subplots(figsize=(8, 6))
for i, slot in enumerate(y.columns[:5]):
    ax1.scatter(y.iloc[:, i], y_pred[:, i], alpha=0.6, label=f"Slot {slot}")
ax1.plot([0, 1], [0, 1], "k--", lw=2)
ax1.set_xlabel("Actual Probability")
ax1.set_ylabel("Predicted Probability")
ax1.set_title("Predicted vs Actual Parking Probabilities")
ax1.legend()
ax1.grid(True)
img1 = fig_to_base64(fig1)
plt.close(fig1)

# F1 score chart
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(f1_scores.keys(), f1_scores.values(), color="#3b82f6")
ax2.set_xticklabels(f1_scores.keys(), rotation=90)
ax2.set_xlabel("Parking Slots")
ax2.set_ylabel("F1 Score")
ax2.set_title("F1 Score per Parking Slot")
plt.tight_layout()
img2 = fig_to_base64(fig2)
plt.close(fig2)

time.sleep(1.5)
spinner.stop("📊 Graphs created successfully")

# ----------------------------
# Step 5: Generating HTML Report
# ----------------------------
spinner = Spinner("Compiling full HTML report")
spinner.start()

html_content = f"""
<html>
<head>
    <title>Parking Model Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f8f9fb;
            margin: 40px;
            color: #1e293b;
        }}
        h1 {{
            color: #1d4ed8;
            font-size: 28px;
        }}
        h2 {{
            color: #334155;
        }}
        .section {{
            background: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            margin-bottom: 25px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #2563eb;
            color: white;
        }}
        img {{
            display: block;
            margin: 20px auto;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        footer {{
            text-align: center;
            color: #6b7280;
            margin-top: 40px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>Parking Occupancy Prediction Model - Evaluation Report</h1>

    <div class="section">
        <h2>1. Model Performance Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Average F1 Score</td><td>{avg_f1:.3f}</td></tr>
            <tr><td>Average Precision</td><td>{avg_precision:.3f}</td></tr>
            <tr><td>Average Recall</td><td>{avg_recall:.3f}</td></tr>
            <tr><td>Mean Squared Error (MSE)</td><td>{mse:.4f}</td></tr>
            <tr><td>Root Mean Squared Error (RMSE)</td><td>{rmse:.4f}</td></tr>
            <tr><td>Mean Absolute Error (MAE)</td><td>{mae:.4f}</td></tr>
            <tr><td>R² Score</td><td>{r2:.4f}</td></tr>
            <tr><td>Explained Variance</td><td>{evs:.4f}</td></tr>
            <tr><td>Overall Accuracy</td><td>{accuracy:.3f}</td></tr>
            <tr><td>Estimated Bias</td><td>{'%.4f' % bias if bias is not None else 'N/A'}</td></tr>
            <tr><td>Estimated Variance</td><td>{'%.4f' % variance if variance is not None else 'N/A'}</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>2. Visual Evaluation</h2>
        <h3>Predicted vs Actual Probabilities</h3>
        <img src="data:image/png;base64,{img1}" width="700">
        <h3>F1 Score Distribution by Slot</h3>
        <img src="data:image/png;base64,{img2}" width="700">
    </div>

    <footer>
        <p>Generated automatically | Machine Learning Model Evaluation Report</p>
    </footer>
</body>
</html>
"""

with open("model_report.html", "w", encoding="utf-8") as f:
    f.write(html_content)
time.sleep(1)
spinner.stop("💾 Report successfully saved as model_report.html")

print("\n✅ Report generation complete.")
print("📊 Open 'model_report.html' in your browser to view the full results.\n")
