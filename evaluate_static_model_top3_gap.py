import pandas as pd
import numpy as np

# Load pre-trained model
model = pd.read_pickle("models/svm_model.pkl")

# Load all data and champion labels
df_all = pd.read_csv("data/Team Summary and Per Game.csv")
df_champs = pd.read_csv("data/champions.csv")
df_champs["Champion"] = 1

# Merge in champion flags
df_all = df_all.merge(df_champs, on=["season", "team"], how="left")
df_all["Champion"] = df_all["Champion"].fillna(0).astype(int)

# Define feature set
features = [
    'o_rtg', 'd_rtg', 'n_rtg',
    'ast_per_game', 'pace',
    'ts_percent', 'e_fg_percent',
    'tov_percent', 'orb_percent',
    'opp_e_fg_percent', 'w', 'l'
]

# Normalize using saved training mean/std
X_all = df_all[features]
X_norm = (X_all - model['mean']) / model['std']
df_all["Margin"] = np.dot(X_norm, model["weights"]) + model["bias"]

# Evaluate each season
results = []
for season, group in df_all.groupby("season"):
    group_sorted = group.sort_values(
        by="Margin", ascending=False).reset_index(drop=True)
    top1 = group_sorted.iloc[0]["team"]
    top3 = group_sorted.head(3)["team"].tolist()

    actual_row = group[group["Champion"] == 1]
    if actual_row.empty:
        continue

    actual_team = actual_row.iloc[0]["team"]
    actual_margin = actual_row.iloc[0]["Margin"]
    top1_margin = group_sorted.iloc[0]["Margin"]
    margin_gap = top1_margin - actual_margin

    results.append({
        "season": season,
        "predicted_champion": top1,
        "top_3": top3,
        "actual_champion": actual_team,
        "correct_top_1": (actual_team == top1),
        "in_top_3": (actual_team in top3),
        "margin_gap": margin_gap
    })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("season_predictions_with_top3_and_gap.csv", index=False)
acc_1 = df_results["correct_top_1"].mean() * 100
acc_3 = df_results["in_top_3"].mean() * 100

print(f"Accuracy (Top 1): {acc_1:.2f}%")
print(f"Accuracy (Top 3): {acc_3:.2f}%")
print(df_results.head(10))
