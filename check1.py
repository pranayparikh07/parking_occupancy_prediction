import numpy as np
import pandas as pd  # <-- Add this
# Assuming pipeline and slot_names are already defined

hours = np.arange(0,24)
days = np.arange(0,7)
grid = pd.DataFrame([(h,d) for h in hours for d in days], columns=["hour","dayofweek"])

# Predict
preds = pipeline.predict(grid)
pred_df = pd.DataFrame(preds, columns=slot_names)
pred_df["hour"] = grid["hour"]
pred_df["dayofweek"] = grid["dayofweek"]
pred_df["avg_availability"] = pred_df[slot_names].mean(axis=1)*100

best_idx = pred_df["avg_availability"].idxmax()
best_row = pred_df.loc[best_idx, ["hour","dayofweek","avg_availability"]]

print(f"✅ Best time for parking:")
print(f"Hour: {int(best_row['hour'])}, Day of Week: {int(best_row['dayofweek'])}, "
      f"Average availability: {best_row['avg_availability']:.2f}%")
