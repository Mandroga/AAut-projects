from imports3 import *

with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(X):
    y_pred = model.predict(X)
    return y_pred
