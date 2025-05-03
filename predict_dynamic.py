import pandas as pd
import numpy as np
from svm_model import LinearSVM

# === Config ===
input_csv = "data/2025_season.csv"
model_path = "models/svm_model.pkl"

# Load model
model_info = pd.read_pickle(model_path)

# Define features used in model
features = [
    'o_rtg', 'd_rtg', 'n_rtg',
    'ast_per_game', 'pace',
    'ts_percent', 'e_fg_percent',
    'tov_percent', 'orb_percent',
    'opp_e_fg_percent', 'w', 'l'
]

# Load and normalize input data
df = pd.read_csv(input_csv)
X = df[features].apply(pd.to_numeric, errors='coerce')
X = (X - model_info['mean']) / model_info['std']

# Load model weights
svm = LinearSVM()
svm.w = model_info['weights']
svm.b = model_info['bias']

# Compute margins
margins = np.dot(X.values, svm.w) + svm.b
threshold = np.percentile(margins, 75)  # Use 3rd quartile as dynamic cutoff

# Predict based on dynamic threshold
df['Margin'] = margins
df['Prediction'] = (margins > threshold).astype(int)

# Sort by descending margin
df_sorted = df.sort_values(by="Margin", ascending=False)

# Print summary
print(f"\n[INFO] Dynamic threshold (75th percentile): {threshold:.2f}")
print("\nTop Predicted Teams:")
print(df_sorted[df_sorted['Prediction'] == 1][['team', 'season', 'Margin']])

# Save output
df_sorted.to_csv("predictions/warriorsVbulls.csv", index=False)
print("\n[INFO] Predictions saved")
