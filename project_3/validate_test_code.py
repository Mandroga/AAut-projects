from mymodel import predict
import pickle

with open("Xtrain2.pkl", "rb") as f:
    X = pickle.load(f)

Y_pred= predict(X)
print(Y_pred)
print(Y_pred.shape)