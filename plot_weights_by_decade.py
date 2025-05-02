import pandas as pd
import matplotlib.pyplot as plt

# Define decades and corresponding weight files
decades = ["80s", "90s", "2000s", "2010s"]
weight_files = [f"models/svm_model_{dec}.pkl" for dec in decades]

# Stat labels (should match your model features)
features = [
    'o_rtg', 'd_rtg', 'n_rtg',
    'ast_per_game', 'pace',
    'ts_percent', 'e_fg_percent',
    'tov_percent', 'orb_percent',
    'opp_e_fg_percent', 'w', 'l'
]

# Collect weight vectors
all_weights = {}
for dec, file in zip(decades, weight_files):
    try:
        model = pd.read_pickle(file)
        all_weights[dec] = model['weights']
    except FileNotFoundError:
        print(f"[WARN] Missing weights file: {file}")

# Create DataFrame for plotting
df_weights = pd.DataFrame(all_weights, index=features)

# Optional: exclude specific stats
# ← example: exclude wins, losses, offensive rebounds
excluded_stats = ['w', 'l']
df_weights = df_weights.drop(index=excluded_stats, errors='ignore')

# Plot
plt.figure(figsize=(12, 6))
df_weights.T.plot(kind="bar", figsize=(14, 6), width=0.8)
plt.axhline(0, color="black", linewidth=0.5)
plt.title("SVM Stat Weights by Decade")
plt.ylabel("Weight")
plt.xlabel("Decade")
plt.legend(title="Stat Feature", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()
