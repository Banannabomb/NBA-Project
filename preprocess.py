# preprocess.py

import pandas as pd

feature_cols = [
    'Standing (West/East)',
    'Offensive Rating',
    'Defensive Rating',
    'Net Rating (ORTG - DRTG)',
    'Assists Per Game (APG)',
    'Pace',
    'True Shooting % (TS%)',
    'Effective FG % (eFG%)',
    'Turnover Ratio (TOV%)',
    'Offensive Rebound %',
    'Opponent eFG%',
    'Wins',
    'Losses'
]


def process_record(df):
    df[['Wins', 'Losses']] = df['Record'].str.split('-', expand=True)
    df['Wins'] = pd.to_numeric(df['Wins'], errors='coerce')
    df['Losses'] = pd.to_numeric(df['Losses'], errors='coerce')
    df = df.drop(columns=['Record'])
    return df


def process_standing(df):
    df['Standing (West/East)'] = df['Standing (West/East)'].str.extract(r'(\d+)').astype(float)
    return df


def process_percentages(df):
    percentage_cols = [
        'True Shooting % (TS%)',
        'Effective FG % (eFG%)',
        'Turnover Ratio (TOV%)',
        'Offensive Rebound %',
        'Opponent eFG%'
    ]
    for col in percentage_cols:
        df[col] = df[col].str.rstrip('%').astype(float)
    return df


def normalize_features(X, mean=None, std=None):
    if mean is None or std is None:
        mean = X.mean()
        std = X.std()
    X_norm = (X - mean) / std
    return X_norm, mean, std


def load_and_prepare_training_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Team', 'Year'], errors='ignore')

    df = process_record(df)
    df = process_standing(df)
    df = process_percentages(df)

    df['Label'] = 1  # All historical teams are champions

    df = df.dropna()

    X = df[feature_cols]
    y = df['Label']
    return X, y


def load_current_season_data(file_path, mean, std):
    df = pd.read_csv(file_path)

    # Extract team names to attach after prediction
    team_names = df['Team']

    # Preprocess features as before
    df['Wins'] = df['Record'].apply(lambda x: int(x.split('-')[0]))
    df['Losses'] = df['Record'].apply(lambda x: int(x.split('-')[1]))
    df['Standing (West/East)'] = df['Standing (West/East)'].apply(
        lambda x: int(x[0]) if isinstance(x, str) else x)

    percent_cols = ['True Shooting % (TS%)', 'Effective FG % (eFG%)', 'Turnover Ratio (TOV%)',
                    'Offensive Rebound %', 'Opponent eFG%']
    for col in percent_cols:
        df[col] = df[col].str.rstrip('%').astype(float) / 100.0

    # Drop original columns and irrelevant ones
    df_clean = df.drop(columns=[
        'Year',
        'Team',  # not in input
        'Record'
    ])

    # Normalize using training mean/std
    X = (df_clean - mean) / std

    return df.assign(Team=team_names), X
