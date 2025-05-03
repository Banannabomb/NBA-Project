import pandas as pd
import numpy as np
from svm_model import LinearSVM
from preprocess import normalize_features
import os

# Load full dataset and champions file
df_all = pd.read_csv("data/Team Summary and Per Game.csv")
df_champs = pd.read_csv("data/champions.csv")
df_champs["Champion"] = 1

# Merge champion label
df_all = df_all.merge(df_champs, on=["season", "team"], how="left")
df_all["Champion"] = df_all["Champion"].fillna(0).astype(int)
df_all["Champion"] = df_all["Champion"].replace(0, -1)

# List of seasons to train on
seasons = sorted(df_all["season"].unique())
features = [
    'o_rtg', 'd_rtg', 'n_rtg',
    'ast_per_game', 'pace',
    'ts_percent', 'e_fg_percent',
    'tov_percent', 'orb_percent',
    'opp_e_fg_percent', 'w', 'l'
]

output_dir = "yearly_weights"
os.makedirs(output_dir, exist_ok=True)

# Train model per year and save weights
for season in seasons:
    df_season = df_all[df_all["season"] == season].copy()

    if df_season["Champion"].nunique() != 2:
        print(f"[WARN] Skipping {season}: missing full label set")
        continue

    X = df_season[features]
    X_norm, _, mean_std = normalize_features(df_season, fit=True)
    y = df_season["Champion"].values

    model = LinearSVM(learning_rate=0.001, lambda_param=0.1, n_iters=3000)
    model.fit(X_norm, y)

    model_info = {
        "season": season,
        "weights": model.w,
        "bias": model.b
    }
    pd.to_pickle(model_info, f"{output_dir}/weights_{season}.pkl")
    print(f"[INFO] Trained and saved model for {season}")
