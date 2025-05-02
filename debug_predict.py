import pandas as pd
import numpy as np
from svm_model import LinearSVM

# === Config ===
input_csv = "data/2025_season.csv"  # Update if needed
model_path = "svm_model.pkl"

# === Load model and input ===
model_info = pd.read_pickle(model_path)

# Confirm saved mean/std are reasonable
print("\n[INFO] Mean values used for normalization:")
print(model_info["mean"])

print("\n[INFO] Std dev values used for normalization:")
print(model_info["std"])

# Define expected features
features = [
    'o_rtg', 'd_rtg', 'n_rtg',
    'ast_per_game', 'pace',
    'ts_percent', 'e_fg_percent',
    'tov_percent', 'orb_percent',
    'opp_e_fg_percent', 'w', 'l'
]

# Load prediction dataset
df = pd.read_csv(input_csv)

# Make sure all feature columns exist and are numeric
for col in features:
    if col not in df.columns:
        raise ValueError(f"Missing expected feature: {col}")
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaNs
df.dropna(subset=features, inplace=True)

# Normalize
X = df[features]
X_norm = (X - model_info['mean']) / model_info['std']

print("\n[DEBUG] Normalized feature sample (first 5 rows):")
print(X_norm.head())

# Restore model weights and predict
svm = LinearSVM()
svm.w = model_info['weights']
svm.b = model_info['bias']

predictions = svm.predict(X_norm.values)
margins = np.dot(X_norm.values, svm.w) + svm.b

# Show debug outputs
df['Prediction'] = predictions
df['Margin'] = margins

print("\n[RESULT] Prediction margin stats:")
print(df['Margin'].describe())

print("\n[RESULT] Teams with highest margins:")
print(df[['team', 'Margin']].sort_values(by='Margin', ascending=False).head(5))

# Optional: Save debug version
df.to_csv("debug_predictions.csv", index=False)
print("\n[INFO] Debug output saved to 'debug_predictions.csv'")

print("\\n[DEBUG] Model weights:")
print(model_info["weights"])
print("\\n[DEBUG] Bias term:")
print(model_info["bias"])
