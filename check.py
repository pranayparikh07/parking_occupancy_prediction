import pandas as pd

# Load data
df = pd.read_csv("SPSIRDATA.csv")
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["hour"] = df["created_at"].dt.hour
df["dayofweek"] = df["created_at"].dt.dayofweek

# Average availability per slot for each hour/day-of-week
prob_table = (
    df.groupby(["hour", "dayofweek", "field1"])["field2"]
      .mean()  # average availability (0-1)
      .reset_index()
)

# Pivot to wide format: each slot is a column
pivot_df = prob_table.pivot_table(
    index=["hour", "dayofweek"],
    columns="field1",
    values="field2"
).fillna(0).reset_index()

print("✅ Aggregated training table created.")
