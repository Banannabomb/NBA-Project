import pandas as pd


def clean_and_merge_data():
    # Load both datasets without misinterpreting "na"
    df_sum = pd.read_csv("data/Team Summaries.csv",
                         keep_default_na=False, na_values=["NA"])
    df_per = pd.read_csv("data/Team Stats Per Game.csv",
                         keep_default_na=False, na_values=["NA"])

    # Merge on team and season
    merged_df = pd.merge(df_sum, df_per, on=["team", "season"], how="left")

    # Select only relevant stat columns
    required_features = [
        'o_rtg', 'd_rtg', 'n_rtg',
        'ast_per_game', 'pace',
        'ts_percent', 'e_fg_percent',
        'tov_percent', 'orb_percent',
        'opp_e_fg_percent', 'w', 'l'
    ]

    # Keep only these + metadata
    keep_cols = ['team', 'season'] + \
        [col for col in required_features if col in merged_df.columns]
    filtered_df = merged_df[keep_cols]

    # Drop rows with missing values only in the selected stat columns
    filtered_df.dropna(subset=required_features, inplace=True)

    # Save final result
    filtered_df.to_csv(
        "data/Improved Team Summary and Per Game.csv", index=False)
    print(f"[INFO] Cleaned data saved with {len(filtered_df)} rows.")


if __name__ == "__main__":
    clean_and_merge_data()
