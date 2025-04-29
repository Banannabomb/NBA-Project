# predict.py

import pandas as pd
import numpy as np
from preprocess import load_current_season_data
from svm_model import LinearSVM

# Load model info
model_info = pd.read_pickle('svm_model.pkl')

# Load and preprocess current season data
df, X_current_norm = load_current_season_data(
    'data/current_season.csv', model_info['mean'], model_info['std'])

# Initialize model and load parameters
svm = LinearSVM()
svm.w = model_info['weights']
svm.b = model_info['bias']

# Make predictions
predictions = svm.predict(X_current_norm.values)
margins = np.dot(X_current_norm.values, svm.w) + svm.b

# Attach predictions and margins to original dataframe (with team name)
df['Prediction'] = predictions
df['Margin'] = margins
df['Team'] = df['Team']  # In case needed explicitly

# Show predictions
print(df[['Team', 'Prediction', 'Margin']])
print("\nTeams Predicted to be Champions:")
print(df[df['Prediction'] == 1])

# Save results
df.to_csv('predictions.csv', index=False)
