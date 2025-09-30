# mymodel.py
import pickle
import numpy as np
from classes import RBFFeatures, DropHighlyCorrelated, DropLowTargetCorrelation


_model_path = "our_best_model.pkl"
with open(_model_path, "rb") as f:
    _pipeline = pickle.load(f)

print(_pipeline.named_steps["scaler"])

def predict(X_test):

    X_arr = np.asarray(X_test)
    preds = _pipeline.predict(X_arr)
    return preds
