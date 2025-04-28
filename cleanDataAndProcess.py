import pandas as pd


def load_and_preprocess(path, drop_label=None, mean=None, std=None):
    df = pd.read_csv(path)

    if drop_label:
        y = df[drop_label].values
        X = df.drop(columns=[drop_label])
    else:
        y = None
        X = df

    if mean is None or std is None:
        mean = X.mean()
        std = X.std()

    X_norm = (X - mean) / std

    return X_norm, y, mean, std
