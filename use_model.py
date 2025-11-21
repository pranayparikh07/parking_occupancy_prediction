# use_model.py
import pandas as pd
import joblib

# ----------------------------
# 1. Load the trained model
# ----------------------------
pipeline, slot_names = joblib.load("parking_prob_model.pkl")
print("✅ Model loaded successfully.")

# ----------------------------
# 2. Example input
# ----------------------------
# You can enter fractional hours like 10.1, 10.25, etc.
hour = 10.3        # Fractional hour of the day
dayofweek = 0      # 0 = Monday, 6 = Sunday

X_new = pd.DataFrame([[hour, dayofweek]], columns=["hour", "dayofweek"])

# ----------------------------
# 3. Predict probabilities
# ----------------------------
preds = pipeline.predict(X_new)[0]  # array of slot probabilities

slot_probs = {slot: prob for slot, prob in zip(slot_names, preds)}

# Sort slots by availability probability
sorted_slots = dict(sorted(slot_probs.items(), key=lambda x: x[1], reverse=True))

# Estimate expected number of free slots
expected_free = sum(preds)

# ----------------------------
# 4. Display results
# ----------------------------
print(f"\nPrediction for hour={hour}, dayofweek={dayofweek}")
print(f"Total slots: {len(slot_names)}")
print(f"Expected available slots: {expected_free:.2f}\n")

print("Top 10 slots likely to be free:")
for slot, prob in list(sorted_slots.items())[:10]:
    print(f"{slot}: {prob*100:.2f}% chance free")
