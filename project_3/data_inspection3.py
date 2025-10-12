# %% imports
from imports3 import *
# %% load data
print(os.getcwd())
with open("Xtrain2.pkl", "rb") as f:
    X = pickle.load(f)
Y= np.load("Ytrain2.npy")

print(X.shape, Y.shape)
# %%
