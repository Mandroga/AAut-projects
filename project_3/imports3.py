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
import myMLtools
importlib.reload(myMLtools) 
from myMLtools import *
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import trim_mean
from sklearn.base import BaseEstimator, TransformerMixin, clone
import joblib
from sklearn.ensemble import IsolationForest    
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from scipy.spatial.distance import euclidean
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import random
from skopt.space import Space
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasClassifier
from keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
import optuna
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, balanced_accuracy_score
from catboost import CatBoostClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression

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


# %% project functions
def make_cols(indexes, components=['x','y']):
    col_names = [c+str(i) for i in indexes for c in components]
    return col_names
def skeleton_sequence_to_df(X_ss):
    col_names = make_cols(range(33))
    df = pd.DataFrame(X_ss, columns=col_names)
    return df

def df_distances(df, indexes):
    df = df.copy()
    diffs = df.diff().fillna(0)
    for i in indexes:
        x, y = f'x{i}', f'y{i}'
        dist = np.hypot(diffs[x], diffs[y])  # faster and clearer than sqrt(x^2 + y^2)
        df[f'dist{i}'] = dist

    return df

def sliding_average(a, window):
    # compute rolling mean along axis=0 (frames)
    cumsum = np.pad(np.cumsum(a, axis=0), ((window,0),(0,0),(0,0)), mode='constant')
    return (cumsum[window:] - cumsum[:-window]) / window

class StratifiedGroupKFoldStrict(BaseCrossValidator):
    def __init__(self, n_splits=3, shuffle=True, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("`groups` must be provided for StratifiedGroupKFoldStrict.")
        y = np.asarray(y)
        groups = np.asarray(groups)

        # Map samples -> group ids 0..G-1
        uniq_groups, inv = np.unique(groups, return_inverse=True)

        # One label per group (majority label inside the group)
        group_labels = np.empty(len(uniq_groups), dtype=y.dtype)
        for gi in range(len(uniq_groups)):
            cls, cnt = np.unique(y[inv == gi], return_counts=True)
            group_labels[gi] = cls[np.argmax(cnt)]

        # Feasibility: each class must have >= n_splits groups
        for cls in np.unique(group_labels):
            if (group_labels == cls).sum() < self.n_splits:
                raise ValueError(
                    f"Not enough groups of class {cls} for {self.n_splits} folds."
                )

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        for g_tr, g_te in skf.split(uniq_groups, group_labels):
            tr_idx = np.where(np.isin(inv, g_tr))[0]
            te_idx = np.where(np.isin(inv, g_te))[0]
            yield tr_idx, te_idx

# %%

def sum_consecutive_distances(points):
        """
        points: array-like of shape (T, D) where D is 1 (e.g. x only) or >1 (e.g. x,y).
        Returns the sum of Euclidean distances between consecutive points:
            sum_{t=0..T-2} ||points[t+1] - points[t]||.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        if pts.shape[0] < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)            # shape (T-1, D)
        if pts.shape[1] == 1:
            return float(np.sum(np.abs(diffs[:, 0])))   # 1D -> absolute diffs
        return float(np.sum(np.linalg.norm(diffs, axis=1)))  # Euclidean per step

def orientationchange(points):
        """
        points: array-like of shape (T, D) where D is 1 (e.g. x only) or >1 (e.g. x,y).
        Returns the number of times the orientation of the movement vector changes significantly.
        A significant change is defined as a change in angle greater than 45 degrees.:
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        if pts.shape[0] < 2:
            return 0.0
        vectors = np.diff(pts, axis=0)            # shape (T-1, D)
        orientations = []
        for vec in vectors:
            norm = np.linalg.norm(vec)
            if norm == 0:
                orientations.append(0)
            else:
                unit_vec = vec / norm
                angle = np.arctan2(unit_vec[1], unit_vec[0])  # angle in radians
                orientations.append(angle)
        orientation_changes = 0
        for i in range(1, len(orientations)):
            angle_diff = np.abs(orientations[i] - orientations[i-1])
            if angle_diff > np.pi / 4:  # greater than 45 degrees
                orientation_changes += 1
        return orientation_changes