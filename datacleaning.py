import pandas as pd
df = pd.read_csv("data/Team Summaries.csv")

df = df.dropna()

df.to_csv('nba_more_data.csv', index=False)
