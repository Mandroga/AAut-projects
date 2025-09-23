# %% import libraries

   # %matplotlib widget
   # %matplotlib qt
if 0:
    #%matplotlib inline
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 200
else:
    import matplotlib
    matplotlib.use("Qt5Agg")
  #  %matplotlib notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import copy
import importlib
import tools
importlib.reload(tools) 
from tools import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import Ridge, Lasso
import itertools
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cross_decomposition import PLSRegression
from skopt import BayesSearchCV
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import trim_mean

# %% import data

X = np.load('X_train.npy')
y= np.load('y_train.npy')

X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)
X_df.columns = X_df.columns.astype(str)

df = X_df.copy()
df['target'] = y_df


x_scaler = StandardScaler()
y_scaler = StandardScaler()

df_scaled = df.copy()
feature_names = [col for col in df.columns if col != 'target']
df_scaled[feature_names] = x_scaler.fit_transform(df_scaled[feature_names])
df_scaled['target'] = y_scaler.fit_transform(df_scaled[['target']])

random_state = 42