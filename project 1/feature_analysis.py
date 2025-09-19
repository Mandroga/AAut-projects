# %% import libraries
if 0:
    %matplotlib widget
elif 0:
    %matplotlib qt
elif 0:
    %matplotlib inline
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

# %% Feature analysis

# %% Feature selection
'''
Three types of methods: Filter, embedded and wrapper 
We will use filter and wrapper

MI gives us information on non linear relations with target (?)
Corr gives us linear relation with target

High MI, low Corr -> different degree poly needed !?
We want features with high correlation !
'''

#Filter methods - Corr with target, Var threshold, MI with target
degree = 6


x_scaler = StandardScaler()
X_scaled = pd.DataFrame(x_scaler.fit_transform(X))
X_scaled.columns = X_scaled.columns.astype(str)

poly = PolynomialFeatures(degree=degree, include_bias=False)

X_poly = pd.DataFrame(poly.fit_transform(X_df))
feature_names = poly.get_feature_names_out(X_df.columns)
X_poly.columns = feature_names

X_scaled_poly = pd.DataFrame(poly.fit_transform(X_scaled))
feature_names = poly.get_feature_names_out(X_scaled.columns)
X_scaled_poly.columns = feature_names

poly_scaler = StandardScaler()
X_sps = pd.DataFrame(poly_scaler.fit_transform(X_scaled_poly), columns=X_scaled_poly.columns)



# %% var thresh
# Var is linear, so for poly means we only need first features
cv = (X_df.std()/X_df.mean().abs())
print(cv.sort_values())
# smallest std is 10% of mean, no features dropped

# %% MI
# Data leakage from using whole dataset for feature selection ?!?!
mi = mutual_info_regression(X_df, y_df.iloc[:,0])
mi_series = pd.Series(mi, index=X_df.columns).sort_values(ascending=False)
print("Mutual Information (top features):")
print(mi_series)

#MI for feature combinations
def MI_poly_feature_comb(X_poly, feature):
    features_filtered = [col for col in X_poly.columns if feature in col and '^' not in col]
    X_poly_MIf = X_poly[features_filtered].copy().drop(feature, axis=1)
    features = X_poly_MIf.columns
    features_nofeat = [" ".join([n for n in feat.split(' ') if n != feature]) for feat in features]
    mi_poly = mutual_info_regression(X_poly_MIf, y_df.iloc[:,0])-mutual_info_regression(X_poly[features_nofeat], y_df.iloc[:,0])
    mi_poly_series = pd.Series(mi_poly, index=X_poly_MIf.columns).sort_values(ascending=False)
    return mi_poly_series

def plot_f(subplot, n):
    mi_poly_series = MI_poly_feature_comb(X_poly, str(n))
    sns.histplot(mi_poly_series, ax=subplot)
    subplot.set_title(n)

min_multiple_plot(6, plot_f)
plt.show()
'''
Feature 1 provides significantly less information comparing to others
Some feature combinations contribute to less MI others More
Features 0 and 3 have more counts on the positive side than others
meaning their combinations provide more information
the opposite happens for 2 and less intensely for 4 and 5
Analyze more!
'''
# %% Corr
corr_with_target = X_poly.corrwith(y_df.iloc[:,0]).abs()

print('#Features corr > threshold')
corr_threshold = [0.01, 0.05, 0.1, 1]
for ct in corr_threshold:
    n_features = (corr_with_target < ct).sum()
    print(f'corr < {ct}: {n_features}')
    
sns.histplot(corr_with_target.values, bins=np.arange(0,1.05,0.05))
plt.grid(True)
plt.show()
print(corr_with_target.sort_values().head())
'''
Very low corr features
Analyze!!!
'''
# %%
