import pandas as pd
df_sum = pd.read_csv("data/Team Summaries.csv", na_values='NA')
df_per = pd.read_csv("data/Team Stats Per Game.csv", na_values='NA')

merged_df = pd.merge(df_sum, df_per, on=['team', 'season'], how='left')
merged_df = merged_df.dropna()

merged_df.to_csv('Team Summary and Per Game 2.csv', index=False)
