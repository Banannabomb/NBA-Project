import pandas as pd
import numpy as np
from preprocess import load_and_prepare_training_data, normalize_features
from svm_model import LinearSVM

# Load and preprocess training data
df = load_and_prepare_training_data()
X_train_norm, y_train, mean_std = normalize_features(df, fit=True)

# Convert 0 → -1 for SVM labels
y_train = np.where(y_train == 0, -1, y_train)

# Initialize and train SVM
svm = LinearSVM(learning_rate=0.00001, lambda_param=0.01,
                n_iters=10000, negative_weight=5.0)

svm.fit(X_train_norm, y_train)

# Save model parameters
model_info = {
    'mean': mean_std['mean'],
    'std': mean_std['std'],
    'weights': svm.w,
    'bias': svm.b
}
pd.to_pickle(model_info, 'models/svm_model_2010s.pkl')
df.to_csv("processed.csv")
print("Training complete. Model saved.")
