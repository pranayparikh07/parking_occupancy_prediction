import pandas as pd
import joblib

# Load
pipeline, slot_names = joblib.load("parking_prob_model.pkl")

# Example input
hour = 10         # hour of day
dayofweek = 0     # Monday
X_new = pd.DataFrame([[hour, dayofweek]], columns=["hour","dayofweek"])

# Predict availability probabilities
proba = pipeline.predict(X_new)[0]  # array of probabilities per slot

slot_free_prob = {name: p*100 for name,p in zip(slot_names, proba)}

# Sort slots
sorted_slots = dict(sorted(slot_free_prob.items(), key=lambda x: x[1], reverse=True))

# Show top 10
print(f"At hour {hour}, day {dayofweek} (0=Monday):")
for slot, prob in list(sorted_slots.items())[:10]:
    print(f"{slot}: {prob:.2f}% chance free")
