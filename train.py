# train.py

import pandas as pd
from preprocess import load_and_prepare_training_data, normalize_features
from svm_model import LinearSVM

# Load and preprocess training data
X_train, y_train = load_and_prepare_training_data('data/nba_data.csv')

# Normalize training features
X_train_norm, mean, std = normalize_features(X_train)

# Initialize and train SVM
svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=10000)
svm.fit(X_train_norm.values, y_train.values)

# Save mean, std, and model weights
model_info = {
    'mean': mean,
    'std': std,
    'weights': svm.w,
    'bias': svm.b
}
pd.to_pickle(model_info, 'svm_model.pkl')

print("Training complete. Model saved as 'svm_model.pkl'.")
