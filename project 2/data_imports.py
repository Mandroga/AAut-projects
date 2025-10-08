# %% imports
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cross_decomposition import PLSRegression
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import ParameterGrid
from scipy.stats import trim_mean
from sklearn.base import BaseEstimator, TransformerMixin, clone
import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest    
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import os
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial.distance import euclidean
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
RANDOM_STATE = 42


# %% useful functions

def winsorized_mape(y_true, y_pred, q=0.95):
    errors = np.abs((y_true - y_pred) / y_true)
    threshold = np.quantile(errors, q)  # cap top q% of errors
    errors = np.clip(errors, 0, threshold)
    return errors.mean()

def score_preds_cv(X, y, model_tup, n_splits=5):
    score_df = pd.DataFrame(columns=['model','metric','fold','set','score'])
    preds = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    model_name, model = model_tup
    iter_name = model_name
    preds[iter_name] = {'train_p':[],'train_t':[],'val_p':[],'val_t':[]}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Fit and predict
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        preds[iter_name]['train_p'].append(y_pred_train)
        preds[iter_name]['val_p'].append(y_pred_val)
        preds[iter_name]['train_t'].append(y_train)
        preds[iter_name]['val_t'].append(y_val)

        # Compute metrics
        sets = [['train_p', y_train, y_pred_train], ['val_p', y_val, y_pred_val]]
        for set_name, truth, pred in sets:
            for i, metric in enumerate(metrics):
                score_df.loc[len(score_df)] = [
                    model_name, metric_names[i], fold+1, set_name, metric(truth, pred)
                ]

    # ------- Summary: mean ± std per metric per model -------
    score_df = score_df.groupby(['model','metric','set']).score.agg(['mean']).reset_index()
    score_df['score']=score_df['mean']
    score_df.drop('mean',axis=1,inplace=True)
    return score_df, preds

def score_preds_grid_cv(X, y, grid, n_splits=5):
    score_df = pd.DataFrame(columns=['model','metric','fold','set','score'])
    preds = {}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for ps in grid:
        model_name, model = ps['model']

        iter_name = model_name
        preds[iter_name] = {'train_p':[],'train_t':[],'val_p':[],'val_t':[]}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)

            preds[iter_name]['train_p'].append(y_pred_train)
            preds[iter_name]['val_p'].append(y_pred_val)
            preds[iter_name]['train_t'].append(y_train)
            preds[iter_name]['val_t'].append(y_val)

            # Compute metrics
            sets = [['train_p', y_train, y_pred_train], ['val_p', y_val, y_pred_val]]
            for set_name, truth, pred in sets:
                for i, metric in enumerate(metrics):
                    score_df.loc[len(score_df)] = [
                        model_name, metric_names[i], fold+1, set_name, metric(truth, pred)
                    ]

    # ------- Summary: mean ± std per metric per model -------
    score_df = score_df.groupby(['model','metric','set']).score.agg(['mean']).reset_index()
    score_df['score']=score_df['mean']
    score_df.drop('mean',axis=1,inplace=True)
    return score_df, preds

def score_preds_grid_tts(X, y, score_df, preds, grid, test_size=0.2):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    for ps in grid:
        model_name, model = ps['model']
        iter_name = model_name
        
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        preds[iter_name] = {'train_p':[],'train_t':[],'val_p':[],'val_t':[]}
        preds[iter_name]['train_p'].append(y_pred_train)
        preds[iter_name]['val_p'].append(y_pred_val)
        preds[iter_name]['train_t'].append(y_train)
        preds[iter_name]['val_t'].append(y_val)
        
        sets = [['train_p',y_train, y_pred_train], ['val_p', y_val, y_pred_val]]
        for set_name, truth, pred in sets:
            for i in range(len(metrics)):
                score_df.loc[len(score_df)] = [model_name, metric_names[i],set_name, metrics[i](truth, pred)]
    return score_df, preds

def overfit_table(df1, df2, metric_names, df_names=['df1', 'df2']):
    # Pivot both dfs
    df1_p = df1.pivot(index=['model', 'metric'], columns='set', values='score')
    df2_p = df2.pivot(index=['model', 'metric'], columns='set', values='score')
    
    # Compute differences
    df1_diff = df1_p['train_p'] - df1_p['val_p']
    df2_diff = df2_p['train_p'] - df2_p['val_p']
    
    # Put them into a dict
    d = {df_names[0]: df1_diff, df_names[1]: df2_diff}
    
    # Create output DataFrame
    overfit_df = pd.DataFrame(columns=metric_names, index=df_names)
    
    for key, df in d.items():
        for metric in metric_names:
            # Select all values for this metric across models
            vals = df.xs(metric, level='metric')
            overfit_df.loc[key, metric] = trim_mean(vals, 0.1)
    
    return overfit_df

class EstimatorWrapper(BaseEstimator, TransformerMixin):
    """Make any estimator appear as a single, non-iterable object to skopt/NumPy."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def transform(self, X):
        return self.estimator_.transform(X)

    # keep it sklearn-friendly
    def get_params(self, deep=True):
        return {"estimator": self.estimator}

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params["estimator"]
        return self
    
class DataSizeLogger(BaseEstimator, TransformerMixin):
    def __init__(self): self.logs_ = []
    def fit(self, X, y=None): return self
    def transform(self, X):
        msg = f"[DataSizeLogger] Samples: {X.shape[0]}, Features: {X.shape[1]}"
        self.logs_.append(msg)
        print(msg)  # shows when n_jobs=1
        return X

class FeatureLogger(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self
    def transform(self, X):
        print(f"[FeatureLogger] Input features: {self.n_features_in_}, Output features: {X.shape[1]}")
        return X

# %% metrics
if 1:    
    metrics = [mean_squared_error, mean_absolute_percentage_error, winsorized_mape, r2_score]
    metric_names = ['MSE', 'MAPE', 'wMAPE','R2']   

# %% data
with open("Xtrain1.pkl", "rb") as f:
    X = pickle.load(f)
Y= np.load("Ytrain1.npy")
df = pd.DataFrame(X)
df['target'] = Y

df2 = X.copy()
df2['target'] = Y
# %%
