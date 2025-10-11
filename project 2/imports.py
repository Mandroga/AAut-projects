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

def f1_macro_loss(y_true, y_pred, sample_weight):
    # y_pred is probs for multi:softprob; turn into labels
    y_pred = np.asarray(y_pred)
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)
    # return a *loss* (lower is better) since older XGB minimizes eval_metric
    return 1.0 - f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight)


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

class FeatureTransform(BaseEstimator, TransformerMixin):
    """
    Adds:
      1) Hand-averaged features: for each hand key ('left','right'),
         creates xmean{key}_hand, ymean{key}_hand, xstd{key}_hand, ystd{key}_hand
      2) Diff features for given pairs: creates {component}diff{a}-{b}
         where 'a' or 'b' can be an int index (e.g., 25) or 'left_hand'/'right_hand'
      3) Aggregate stdevs and torso length; returns only engineered columns.
    """

    def __init__(self):
        self.keypoint_side = {
            'left':  [4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
            'right': [1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31]
        }
        self.keypoints_hands = {'left': [15,17,19,21], 'right': [16,18,20,22]}
        self.diff_pairs = ((25,23), (26,24), ('left_hand', 0), ('right_hand', 0),(13,0),(14,0))
        self.components_hand = ('xmean','ymean','xstd','ystd')
        self.components_diff = ('xmean','ymean')

        # will be set in fit
        self.input_features_ = None
        self.output_features_ = None

    def _make_cols(self, indexes, components):
        return [f"{comp}{idx}" for comp in components for idx in indexes]

    def _hand_feature_names(self):
        return [f"{comp}{key}_hand"
                for key in self.keypoints_hands.keys()
                for comp in self.components_hand]

    def _diff_feature_names(self):
        return [f"{comp}diff{a}-{b}"
                for a, b in self.diff_pairs
                for comp in self.components_diff]

    def _knee_std_cols(self):
        return [f"{c}{i}" for i in (25, 26) for c in ('xstd','ystd')]

    def fit(self, X, y=None):
        # record input feature names if provided
        self.input_features_ = list(X.columns) if isinstance(X, pd.DataFrame) else None

        # declare output feature names deterministically (what transform will return)
        hand_cols  = self._hand_feature_names()
        diff_cols  = self._diff_feature_names()
        knee_cols  = self._knee_std_cols()
        std_cols   = ['left_std', 'right_std']
        others     = ['torso_length']
        self.output_features_ = hand_cols + diff_cols + knee_cols + std_cols + others
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            # best-effort: use recorded input column names if we have them
            X = pd.DataFrame(X, columns=self.input_features_)
        X = X.copy()

        # --- 1) Hand averages ---
        if 1:
            hand_cols_out = []
            for key, indexes in self.keypoints_hands.items():
                cols_all = self._make_cols(indexes, self.components_hand)
                for comp in self.components_hand:
                    sub_cols = [c for c in cols_all if c.startswith(comp) and c in X.columns]
                    out_col = f"{comp}{key}_hand"
                    if sub_cols:
                        X[out_col] = X[sub_cols].mean(axis=1)
                    else:
                        X[out_col] = np.nan
                    hand_cols_out.append(out_col)
        #
        # --- 2) Torso length (average of left/right shoulder-to-hip diagonals) ---
        if 1:
            X['torso_length'] = np.sqrt((X.get('xmean11') - X.get('xmean23'))**2 +
                                        (X.get('ymean11') - X.get('ymean23'))**2) / 2.0 \
                                + np.sqrt((X.get('xmean12') - X.get('xmean24'))**2 +
                                        (X.get('ymean12') - X.get('ymean24'))**2) / 2.0

        # --- 3) Diffs (xmean/ymean only) ---
        if 1:
            diff_cols_out = []
            for a, b in self.diff_pairs:
                for comp in self.components_diff:
                    c1 = f"{comp}{a}"
                    c2 = f"{comp}{b}"
                    out_col = f"{comp}diff{a}-{b}"
                    if c1 in X.columns and c2 in X.columns:
                        X[out_col] = (X[c1] - X[c2])/X['torso_length']
                    else:
                        X[out_col] = np.nan
                    diff_cols_out.append(out_col)

        # --- 4) Aggregate stds (left/right) ---
        if 1:
            left_cols  = [c for c in self._make_cols(self.keypoint_side['left'],  ['xstd','ystd']) if c in X.columns]
            right_cols = [c for c in self._make_cols(self.keypoint_side['right'], ['xstd','ystd']) if c in X.columns]
            X['left_std']  = X[left_cols].sum(axis=1)  if left_cols  else np.nan
            X['right_std'] = X[right_cols].sum(axis=1) if right_cols else np.nan
            std_cols_out = ['left_std','right_std']

 
        # --- 5) Knee std cols passthrough (ensure they exist; fill NaN if missing) ---
        knee_std_cols = self._knee_std_cols()
        for col in knee_std_cols:
            if col not in X.columns:
                X[col] = np.nan

        # order the output columns deterministically
        out_cols = hand_cols_out + diff_cols_out + knee_std_cols + std_cols_out + ['torso_length']
        print(out_cols)
        return X[out_cols].to_numpy()

    def get_feature_names_out(self, input_features=None):
        # reflect exactly what transform outputs
        if self.output_features_ is not None:
            return np.array(self.output_features_, dtype=object)
        # fallback
        return np.array(input_features if input_features is not None else [], dtype=object)# %% metrics

class FeatureTransform_np(BaseEstimator, TransformerMixin):
    """
    Input X: numpy array with columns in the order:
      [xmean0, ymean0, xmean1, ymean1, ..., xmean32, ymean32,
       xstd0,  ystd0,  xstd1,  ystd1,  ..., xstd32,  ystd32]

    Adds:
      1) Hand-averaged features for ('left','right') across indices in self.keypoints_hands
         -> xmean{key}_hand, ymean{key}_hand, xstd{key}_hand, ystd{key}_hand
      2) Diff features for given pairs (normalized by torso_length):
         -> {component}diff{a}-{b}  for comp in ('xmean','ymean') and (a,b) in self.diff_pairs
            where a/b ∈ {int joint index, 'left_hand', 'right_hand'}
      3) Aggregate stds over left/right sides: left_std, right_std
      4) Torso length = mean of sqrt((11-23)^2) and sqrt((12-24)^2) using xmean/ymean
      5) Passthrough knee stds: xstd25, ystd25, xstd26, ystd26

    Returns: numpy array with columns in a stable, documented order
    """

    def __init__(self):
        # joint indices (0..32)
        self.keypoint_side = {
            'left':  [4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
            'right': [1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31]
        }
        self.keypoints_hands = {'left': [15,17,19,21], 'right': [16,18,20,22]}
        self.diff_pairs = (
            (25,23), (26,24), (27,23), (28,24),
            ('left_hand', 0), ('right_hand', 0),
            ('left_hand', 11), ('right_hand', 12),
            (13,0), (14,0), (13,11), (14,12)
        )
        self.components_hand = ('xmean','ymean','xstd','ystd')
        self.components_diff = ('xmean','ymean')

        self.n_joints_expected = 33  # 0..32
        self.output_features_ = None

    # --------- helpers on numpy layout ----------
    def _split_blocks(self, X):
        """Return (means_block, stds_block), each shape (n_samples, 2*n_joints)."""
        nJ = self.n_joints_expected
        if X.shape[1] < 4*nJ:
            raise ValueError(
                f"Expected at least {4*nJ} columns (got {X.shape[1]}). "
                "Columns must be [xmean0,ymean0,...,xmean32,ymean32, xstd0,ystd0,...,xstd32,ystd32]."
            )
        means_block = X[:, :2*nJ]
        stds_block  = X[:, 2*nJ:2*nJ+2*nJ]  # keep exactly 2*nJ for stds
        return means_block, stds_block

    def _view_comp(self, X, comp):
        """
        Return view of a component with shape (n_samples, n_joints) under the layout:
        [xmean,ymean]*nJ | [xstd,ystd]*nJ
        """
        nJ = self.n_joints_expected
        means_block, stds_block = self._split_blocks(X)

        if comp == 'xmean':
            return means_block[:, 0::2][:, :nJ]
        elif comp == 'ymean':
            return means_block[:, 1::2][:, :nJ]
        elif comp == 'xstd':
            return stds_block[:, 0::2][:, :nJ]
        elif comp == 'ystd':
            return stds_block[:, 1::2][:, :nJ]
        else:
            raise ValueError(f"Unknown component '{comp}'")

    def _hand_feature_names(self):
        return [f"{comp}{key}_hand"
                for key in ('left','right')
                for comp in self.components_hand]

    def _diff_feature_names(self):
        return [f"{comp}diff{a}-{b}"
                for (a,b) in self.diff_pairs
                for comp in self.components_diff]

    def _knee_std_cols(self):
        return [f"{c}{i}" for i in (25, 26) for c in ('xstd','ystd')]

    def fit(self, X, y=None):
        # validate shape
        nJ = self.n_joints_expected
        if X.ndim != 2 or X.shape[1] < 4*nJ:
            raise ValueError(
                f"Expected at least {4*nJ} columns (got {X.shape[1]}). "
                "Columns must be [xmean0,ymean0,...,xmean32,ymean32, xstd0,ystd0,...,xstd32,ystd32]."
            )

        # define output names (order matches transform stacking)
        hand_cols = self._hand_feature_names()
        diff_cols = self._diff_feature_names()
        knee_cols = self._knee_std_cols()
        std_cols  = ['left_std', 'right_std']
        others    = ['torso_length']
        self.output_features_ = hand_cols + diff_cols + knee_cols + std_cols + others
        return self

    def transform(self, X):
        if X.ndim != 2:
            raise ValueError("X must be 2D NumPy array.")

        # views by component (shape: n_samples x n_joints)
        Xm = self._view_comp(X, 'xmean')
        Ym = self._view_comp(X, 'ymean')
        Xs = self._view_comp(X, 'xstd')
        Ys = self._view_comp(X, 'ystd')

        # 1) Hand averages
        def hand_avg(comp_mat, idxs):
            return comp_mat[:, idxs].mean(axis=1)

        hand_feats = []
        for key, idxs in (('left', self.keypoints_hands['left']),
                          ('right', self.keypoints_hands['right'])):
            hand_feats.extend([
                hand_avg(Xm, idxs),  # xmean{key}_hand
                hand_avg(Ym, idxs),  # ymean{key}_hand
                hand_avg(Xs, idxs),  # xstd{key}_hand
                hand_avg(Ys, idxs),  # ystd{key}_hand
            ])

        hand_dict = {
            'left':  {'xmean': hand_feats[0], 'ymean': hand_feats[1], 'xstd': hand_feats[2], 'ystd': hand_feats[3]},
            'right': {'xmean': hand_feats[4], 'ymean': hand_feats[5], 'xstd': hand_feats[6], 'ystd': hand_feats[7]},
        }

        # 2) Torso length (avg of the two diagonals): joints (11,23) and (12,24)
        left_torso  = np.sqrt((Xm[:,11] - Xm[:,23])**2 + (Ym[:,11] - Ym[:,23])**2)
        right_torso = np.sqrt((Xm[:,12] - Xm[:,24])**2 + (Ym[:,12] - Ym[:,24])**2)
        torso_length = (left_torso + right_torso) / 2.0
        denom = np.where(torso_length == 0.0, np.nan, torso_length)

        # 3) Diffs (xmean/ymean), normalized by torso_length
        def ref_values(comp, ref):
            if isinstance(ref, str):
                if ref == 'left_hand':
                    return hand_dict['left'][comp]
                elif ref == 'right_hand':
                    return hand_dict['right'][comp]
                else:
                    raise ValueError(f"Unknown ref '{ref}'")
            # int joint index
            return (Xm if comp == 'xmean' else Ym)[:, ref]

        diff_feats = []
        for a, b in self.diff_pairs:
            for comp in self.components_diff:
                va = ref_values(comp, a)
                vb = ref_values(comp, b)
                diff_feats.append((va - vb) / denom)

        # 4) Aggregate stds over left/right sides
        left_std  = Xs[:, self.keypoint_side['left']].sum(axis=1) + Ys[:, self.keypoint_side['left']].sum(axis=1)
        right_std = Xs[:, self.keypoint_side['right']].sum(axis=1) + Ys[:, self.keypoint_side['right']].sum(axis=1)

        # 5) Knee std passthrough (xstd25, ystd25, xstd26, ystd26)
        knee_std = [Xs[:,25], Ys[:,25], Xs[:,26], Ys[:,26]]

        # Stack in the documented order
        out_blocks = hand_feats + diff_feats + knee_std + [left_std, right_std] + [torso_length]
        return np.column_stack(out_blocks).astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_features_, dtype=object)#metrics

class FeatureTransform2(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keypoint_side = {'l':[0, 4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
                 'r':[1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31]}
        self.keypoint_nose = 0
        self.keypoint_knees = {'l':26, 'r':25}
        self.keypoints_torso = {'l':[11,23],'r':[12,24]}
        self.keypoints_hands = {'l': [15,17,19,21], 'r': [16,18,20,22]}

        self.x_mean_i = [i for i in range(0, 66, 2)]
        self.y_mean_i = [i for i in range(1, 66, 2)]
        self.x_std_i = [i for i in range(66, 132, 2)]
        self.y_std_i = [i for i in range(67, 132, 2)]

    def _make_cols(self, indexes, components=['xmean','ymean','xstd','ystd']):
        return [txt + str(i) for i in indexes for txt in components]

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        #make df
        col_names = [c+str(i) for i in range(33) for c in ['xmean','ymean']] + [c+str(i) for i in range(33) for c in ['xstd','ystd']]
        df_ft = pd.DataFrame(X, columns=col_names)

        #normalize skeleton
        if 1:
            for key, side_indexes in self.keypoint_side.items():
                x_cols = self._make_cols(side_indexes, components=['xmean','xstd'])
                y_cols = self._make_cols(side_indexes, components=['ymean','ystd'])
                if key == 'l':
                    df_ft.loc[:,x_cols] = df_ft[x_cols].to_numpy() / np.abs(df_ft['xmean11'].to_numpy().reshape(len(df_ft),-1))
                    df_ft.loc[:,y_cols] = df_ft[y_cols].to_numpy() / np.abs(df_ft['ymean11'].to_numpy().reshape(len(df_ft),-1))
                else:
                    df_ft.loc[:,x_cols] = df_ft[x_cols].to_numpy() / np.abs(df_ft['xmean12'].to_numpy().reshape(len(df_ft),-1))
                    df_ft.loc[:,y_cols] = df_ft[y_cols].to_numpy() / np.abs(df_ft['ymean12'].to_numpy().reshape(len(df_ft),-1))
        
        #side std
        if 1:
            for key, side_indexes in self.keypoint_side.items():
                x_cols = self._make_cols(side_indexes, components=['xstd'])
                y_cols = self._make_cols(side_indexes, components=['ystd'])
                df_ft[key + 'std'] = df_ft[x_cols + y_cols].sum(axis=1)


        return df_ft

class FeatureTransform3(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keypoint_side = {'r':[4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
                 'l':[0,1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31]}

        self.x_mean_i = [i for i in range(0, 66, 2)]
        self.y_mean_i = [i for i in range(1, 66, 2)]
        self.x_std_i = [i for i in range(66, 132, 2)]
        self.y_std_i = [i for i in range(67, 132, 2)]

    def _make_cols(self, indexes, components=['xmean','ymean','xstd','ystd']):
        return [txt + str(i) for i in indexes for txt in components]

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        #make df
        col_names = [c+str(i) for i in range(33) for c in ['xmean','ymean']] + [c+str(i) for i in range(33) for c in ['xstd','ystd']]
        df_ft = pd.DataFrame(X, columns=col_names)

        #invert y
        if 1:
            ymean_cols = [col for col in df_ft.columns if 'ymean' in col]
            df_ft.loc[:, ymean_cols] = -1* df_ft[ymean_cols]
        #center skeleton
        if 0:
            xmean_cols = [col for col in df_ft.columns if 'xmean' in col]
            ymean_cols = [col for col in df_ft.columns if 'ymean' in col]
            xtorso_cols = self._make_cols([11,12,23,24], components=['xmean'])
            ytorso_cols = self._make_cols([11,12,23,24], components=['ymean'])
            cx = df_ft[xtorso_cols].mean(axis=1).to_numpy().reshape(len(df_ft),1)
            cy = df_ft[ytorso_cols].mean(axis=1).to_numpy().reshape(len(df_ft),1)
            df_ft.loc[:, xmean_cols] = df_ft[xmean_cols].to_numpy() - cx
            df_ft.loc[:, ymean_cols] = df_ft[ymean_cols].to_numpy() - cy
        #normalize skeleton
        if 0:
            for key, side_indexes in self.keypoint_side.items():
                x_cols = self._make_cols(side_indexes, components=['xmean','xstd'])
                y_cols = self._make_cols(side_indexes, components=['ymean','ystd'])
                if key == 'l':
                    df_ft.loc[:,x_cols] = df_ft[x_cols].to_numpy() / np.abs((df_ft['xmean11']-df_ft['xmean12']).to_numpy().reshape(len(df_ft),-1))
                    df_ft.loc[:,y_cols] = df_ft[y_cols].to_numpy() / np.abs((df_ft['ymean11']-df_ft['ymean23']).to_numpy().reshape(len(df_ft),-1))
                else:
                    df_ft.loc[:,x_cols] = df_ft[x_cols].to_numpy() / np.abs((df_ft['xmean12']-df_ft['xmean11']).to_numpy().reshape(len(df_ft),-1))
                    df_ft.loc[:,y_cols] = df_ft[y_cols].to_numpy() / np.abs((df_ft['ymean12']-df_ft['ymean24']).to_numpy().reshape(len(df_ft),-1))

                #normalize skeleton
        #normalize skeleton2
        if 1:
            xnorm =  np.abs((df_ft['xmean11']-df_ft['xmean12']).to_numpy().reshape(len(df_ft),-1))
            ynorm = 0.5 * (np.abs((df_ft['ymean11']-df_ft['ymean23']).to_numpy().reshape(len(df_ft),-1))+np.abs((df_ft['ymean12']-df_ft['ymean24']).to_numpy().reshape(len(df_ft),-1)))
            for key, side_indexes in self.keypoint_side.items():
                x_cols = self._make_cols(side_indexes, components=['xmean','xstd'])
                y_cols = self._make_cols(side_indexes, components=['ymean','ystd'])
                if key == 'l':
                    df_ft.loc[:,x_cols] = df_ft[x_cols].to_numpy() / xnorm
                    df_ft.loc[:,y_cols] = df_ft[y_cols].to_numpy() / ynorm
                else:
                    df_ft.loc[:,x_cols] = df_ft[x_cols].to_numpy() / xnorm
                    df_ft.loc[:,y_cols] = df_ft[y_cols].to_numpy() / ynorm
        #side std
        if 1:
            for key, side_indexes in self.keypoint_side.items():
                x_cols = self._make_cols(side_indexes, components=['xstd'])
                y_cols = self._make_cols(side_indexes, components=['ystd'])
                df_ft[key + 'std'] = df_ft[x_cols + y_cols].sum(axis=1)
        #only one side
        if 1:
            mask = df_ft['lstd']>df_ft['rstd']
            xmean_cols = [col for col in df_ft.columns if 'xmean' in col]
            df_ft.loc[mask, xmean_cols] = -1* df_ft.loc[mask, xmean_cols]
            lstd = df_ft.loc[mask, 'lstd'].copy()
            rstd = df_ft.loc[mask, 'rstd'].copy()
            df_ft.loc[mask, 'lstd'] = rstd
            df_ft.loc[mask, 'rstd'] = lstd
        
        #hand distance
        if 1:
            df_ft['lhand_dist'] = np.sqrt((df_ft['xmean15'] - df_ft['xmean0'])**2 + (df_ft['ymean15'] - df_ft['ymean0'])**2)
            df_ft['rhand_dist'] = np.sqrt((df_ft['xmean16'] - df_ft['xmean0'])**2 + (df_ft['ymean16'] - df_ft['ymean0'])**2)
        #hip diff
        if 1:
            df_ft['lhip_diff'] = df_ft['xmean25'] - df_ft['xmean23']
            df_ft['rhip_diff'] = df_ft['xmean26'] - df_ft['xmean24']
        
        return df_ft
      
# %%
