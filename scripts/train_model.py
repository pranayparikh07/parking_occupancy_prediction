# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib

# ----------------------------
# 1. Load and prepare data
# ----------------------------
df = pd.read_csv("../data/SPSIRDATA.csv")
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["hour"] = df["created_at"].dt.hour
df["dayofweek"] = df["created_at"].dt.dayofweek

# Compute average availability per slot for each hour/day-of-week
prob_table = (
    df.groupby(["hour", "dayofweek", "field1"])["field2"]
      .mean()  # average availability (probability free)
      .reset_index()
)

# Pivot to wide format: each slot becomes a column
pivot_df = prob_table.pivot_table(
    index=["hour", "dayofweek"],
    columns="field1",
    values="field2"
).fillna(0).reset_index()

print("✅ Aggregated training table created.")

# ----------------------------
# 2. Train model
# ----------------------------
# Features & targets
X = pivot_df[["hour", "dayofweek"]]
y = pivot_df.drop(columns=["hour", "dayofweek"])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline: scale + multioutput regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MultiOutputRegressor(RandomForestRegressor(
        n_estimators=300, random_state=42
    )))
])

pipeline.fit(X_train, y_train)
print("✅ Parking probability model trained successfully.")

# ----------------------------
# 3. Save model
# ----------------------------
slot_names = list(y.columns)
joblib.dump((pipeline, slot_names), "../models/parking_prob_model.pkl")
print("✅ Model saved as models/parking_prob_model.pkl")
