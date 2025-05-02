import pandas as pd


def load_and_prepare_training_data():
    df = pd.read_csv("data/Team Summary and Per Game.csv")

    # Selected features and renamed mapping
    features = [
        'o_rtg', 'd_rtg', 'n_rtg',
        'ast_per_game', 'pace',
        'ts_percent', 'e_fg_percent',
        'tov_percent', 'orb_percent',
        'opp_e_fg_percent', 'w', 'l'
    ]

    df = df[features + ['team', 'season']]

    # Ensure numeric types
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')

    # Load champion list and merge
    champions_df = pd.read_csv("data/champions.csv")
    champions_df["Champion"] = 1
    df = df.merge(champions_df, how="left", on=["season", "team"])
    df["Champion"] = df["Champion"].fillna(0).astype(int)

    # Drop rows with any missing values
    df.dropna(inplace=True)
    return df


def normalize_features(df, fit=True, mean_std=None):
    features = [
        'o_rtg', 'd_rtg', 'n_rtg',
        'ast_per_game', 'pace',
        'ts_percent', 'e_fg_percent',
        'tov_percent', 'orb_percent',
        'opp_e_fg_percent', 'w', 'l'
    ]

    if fit:
        mean_std = {
            'mean': df[features].mean(),
            'std': df[features].std()
        }

    normalized = (df[features] - mean_std['mean']) / mean_std['std']
    labels = df["Champion"].values
    return normalized.values, labels, mean_std
