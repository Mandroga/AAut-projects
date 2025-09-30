import numpy as np
import pickle

# Load your trained pipeline
with open("our_best_model.pkl", "rb") as f:
    pipe = pickle.load(f)

# Generate dummy data
X_test = np.random.rand(5, 6) * 100  # unscaled on purpose

# 1. Raw prediction using pipeline
y_pred_pipeline = pipe.predict(X_test)

# 2. Check intermediate step: what happens after scaling?
X_scaled = pipe.named_steps['scaler'].transform(X_test)

print("Original X_test (first row):", X_test[0])
print("After RobustScaler (first row):", X_scaled[0])
print("Final predictions:", y_pred_pipeline)

# Look at the selected features after preprocessing
X_transformed = pipe[:-1].transform(X_test)  # everything except final regressor
print("Shape after preprocessing:", X_transformed.shape)
print("First row after preprocessing:", X_transformed[0])

# Check regression coefficients
reg = pipe.named_steps['reg']
print("Regression coefficients:", reg.coef_)
print("Regression intercept:", reg.intercept_)
