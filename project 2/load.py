
#%%
%matplotlib inline

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

#%% Load a pickled object
import pickle
path = "Xtrain1.pkl"
with open(path, "rb") as f:
    X = pickle.load(f)
# now `obj` holds whatever was pickled
# %%
print(X)
# %%
