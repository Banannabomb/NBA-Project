import pandas as pd


def extract_seasons(input_csv, years, team=None, output_csv=None):
    """
    Filters the input CSV for rows matching any of the given years (seasons),
    and optionally a specific team name.

    Parameters:
    - input_csv: CSV file to read from
    - years: list or range of seasons (e.g., [2010, 2011] or range(2000, 2010))
    - team: optional team name to filter
    - output_csv: file path to save output
    """
    df = pd.read_csv(input_csv)

    if 'season' not in df.columns:
        raise ValueError("CSV must have a 'season' column.")

    df['season'] = df['season'].astype(str)
    years = [str(y) for y in years]
    filtered_df = df[df['season'].isin(years)]

    if team:
        filtered_df = filtered_df[filtered_df['team'].str.lower(
        ).str.strip() == team.lower().strip()]

    if output_csv:
        filtered_df.to_csv(output_csv, index=False)
        print(f"Filtered data for seasons {years}" + (
            f", team '{team}'" if team else "") + f" saved to '{output_csv}'.")
    else:
        return filtered_df


# Example usage
if __name__ == "__main__":
    extract_seasons("data/Team Summary and Per Game.csv",
                    list(range(2007, 2008)), output_csv="data/2007_season.csv")
