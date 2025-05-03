import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Full list of features
features = [
    'o_rtg', 'd_rtg', 'n_rtg',
    'ast_per_game', 'pace',
    'ts_percent', 'e_fg_percent',
    'tov_percent', 'orb_percent',
    'opp_e_fg_percent', 'w', 'l'
]

# === Exclude stats here if desired ===
excluded = ['ast_per_game',
            'tov_percent', 'orb_percent',
            'opp_e_fg_percent', 'w', 'l', 'n_rtg']
features_to_plot = [f for f in features if f not in excluded]

# Load all yearly weight files
records = []
for file in sorted(os.listdir("yearly_weights")):
    if file.endswith(".pkl"):
        data = pd.read_pickle(os.path.join("yearly_weights", file))
        year = data['season']
        weights = data['weights']
        records.append([year] + list(weights))

df = pd.DataFrame(records, columns=["season"] + features)
df = df.sort_values(by="season")
df["season"] = df["season"].astype(str)

# Plot only selected features
plt.figure(figsize=(14, 6))
for feat in features_to_plot:
    sns.lineplot(data=df, x="season", y=feat, label=feat)

plt.title("SVM Stat Weights by Season (1980–2023)")
plt.xticks(rotation=45)
plt.ylabel("Weight")
plt.xlabel("Season")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
