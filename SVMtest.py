from svm_model import LinearSVM
from preprocess import load_and_preprocess
import numpy as np
import pandas as pd

# Load model checkpoint
checkpoint = np.load('model_checkpoint.npz')
w = checkpoint['w']
b = checkpoint['b']
mean = checkpoint['mean']
std = checkpoint['std']

# Load current season data
X_current, _, _, _ = load_and_preprocess(
    'data/current_top3.csv', mean=pd.Series(mean), std=pd.Series(std))

# Reconstruct model
svm = LinearSVM()
svm.w = w
svm.b = b

# Predict
predictions = svm.predict(X_current.values)
confidences = svm.decision_function(X_current.values)

df_current = pd.read_csv('data/current_top3.csv')
df_current['Predicted_Champion'] = predictions
df_current['Confidence_Score'] = confidences

print(df_current[['Team', 'Predicted_Champion', 'Confidence_Score']])

# Best guess
predicted_winner = df_current.loc[df_current['Confidence_Score'].idxmax()]
print("\nPredicted NBA Champion:", predicted_winner['Team'])
