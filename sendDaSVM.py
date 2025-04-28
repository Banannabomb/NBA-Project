from svm_model import LinearSVM
from preprocess import load_and_preprocess
import numpy as np

# Load training data
X_train, y_train, X_mean, X_std = load_and_preprocess(
    'data/nba_data.csv', drop_label='Champion')

# Train SVM
svm = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X_train.values, y_train)

# Save model and normalization params (simple way)
np.savez('model_checkpoint.npz', w=svm.w, b=svm.b,
         mean=X_mean.values, std=X_std.values)

print("Model trained and saved.")
