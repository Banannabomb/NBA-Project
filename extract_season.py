import pandas as pd


def extract_season(input_csv, year, team=None, output_csv=None):
    """
    Filters the input CSV for rows matching the given season (and optionally a specific team),
    then saves or returns the result.
    """
    df = pd.read_csv(input_csv)

    if 'season' not in df.columns:
        raise ValueError("The CSV must have a 'season' column.")

    # Filter by season
    df = df[df['season'].astype(str) == str(year)]

    # Optional: Filter by team name
    if team:
        df = df[df['team'].str.lower().str.strip() == team.lower().strip()]

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Filtered data for season {year}" + (
            f", team '{team}'" if team else "") + f" saved to '{output_csv}'.")
    else:
        return df


# Example usage
if __name__ == "__main__":
    # You can edit this line to try different filters
    extract_season("data/Team Summary and Per Game.csv",
                   2020, output_csv="data/2020_season.csv")
