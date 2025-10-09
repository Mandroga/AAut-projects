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
from sklearn.model_selection import KFold, cross_val_score, StratifiedGroupKFold, GroupShuffleSplit
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
import random
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
        return X[out_cols]

    def get_feature_names_out(self, input_features=None):
        # reflect exactly what transform outputs
        if self.output_features_ is not None:
            return np.array(self.output_features_, dtype=object)
        # fallback
        return np.array(input_features if input_features is not None else [], dtype=object)# %% metrics

if 1:    
    metrics = [mean_squared_error, mean_absolute_percentage_error, winsorized_mape, r2_score]
    metric_names = ['MSE', 'MAPE', 'wMAPE','R2']   

# %% data
with open("Xtrain1.pkl", "rb") as f:
    X = pickle.load(f)
Y= np.load("Ytrain1.npy")
df = pd.DataFrame(X)
df['target'] = Y

# %% df_ - df unpacked
df_ = df.copy()

x_mean_i = [i for i in range(0, 66, 2)]
y_mean_i = [i for i in range(1, 66, 2)]
x_std_i = [i for i in range(66, 132, 2)]
y_std_i = [i for i in range(67, 132, 2)]
keypoints = list(range(33))
for i in keypoints:
    df_[f'xmean{i}'] = df_['Skeleton_Features'].apply(lambda x: x[x_mean_i[i]])
    df_[f'ymean{i}'] = df_['Skeleton_Features'].apply(lambda x: x[y_mean_i[i]])
    df_[f'xstd{i}'] = df_['Skeleton_Features'].apply(lambda x: x[x_std_i[i]])
    df_[f'ystd{i}'] = df_['Skeleton_Features'].apply(lambda x: x[y_std_i[i]])
df_ = df_.drop(['Skeleton_Features'], axis=1)
print(df_.groupby('Patient_Id')['target'].value_counts().unstack())

# %% keypoint parts
keypoint_side = {'left':[4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
                 'right':[1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31]
                }
keypoints_hands = {'left': [15,17,19,21], 'right': [16,18,20,22]}
def make_cols(indexes, components=['xmean','ymean','xstd','ystd']):
    return [txt + str(i) for i in indexes for txt in components]
side_cols = {k: make_cols(v) for k,v in keypoint_side.items()}

# %% df processed
df_processed = df_.copy()

#impairment side
if 1:
    df_stroke = df_processed.copy()
    for patient_id in range(1,15):
        for key, indexes in keypoint_side.items():
            cols = [txt + str(i) for txt in ['xstd', 'ystd'] for i in indexes]
            mask = df_stroke['Patient_Id']==patient_id
            df_stroke.loc[mask, key + 'std'] = df_stroke[mask][cols].sum().sum()
    df_stroke['impairment_side'] = (df_stroke['leftstd'] > df_stroke['rightstd']).astype(int)
    df_processed['impairment_side'] = df_stroke['impairment_side']

# average hand keypoints for hand feature!
if 1:
    hand_cols = []
    for key, indexes in keypoints_hands.items():
        cols = make_cols(indexes)
        components = ['xmean','ymean','xstd','ystd']
        for component in components:
            sub_cols = [col for col in cols if component in col]
            col_name = f'{component}{key}_hand'
            df_processed[col_name] = df_processed[sub_cols].mean(axis=1)
            hand_cols.append(col_name)
    
# diff cols 
if 1:
    diff_cols = []
    diff_index_list = [(25,23), (26,24), ('left_hand', 9), ('right_hand',10)]
    components = ['xmean','ymean']
    for di1,di2 in diff_index_list:
        for component in components:
            col_name = f'{component}diff{di1}-{di2}'
            df_processed[col_name] = df_processed[f'{component}{di1}'] - df_processed[f'{component}{di2}']
            diff_cols.append(col_name)

# normalize knee std and hand std by torso lenght!
if 1:
    left_torso = np.sqrt((df_['xmean11'] - df_['xmean23'])**2 + (df_['ymean11'] - df_['ymean23'])**2)
    right_torso = np.sqrt((df_['xmean12'] - df_['xmean24'])**2 + (df_['ymean12'] - df_['ymean24'])**2)
    torso_length = (left_torso + right_torso) / 2
   # df_processed['torso_length'] = torso_length

knee_std_cols = [c+str(i) for i in [25,26] for c in ['xstd','ystd']]

df_processed = df_processed[['Patient_Id'] + hand_cols + diff_cols + knee_std_cols + ['impairment_side','target']].copy()

#weights
if 1:
    w = df_processed[[txt+str(j) for txt in ['xstd','ystd'] for j in ['left_hand','right_hand',25,26]]].copy()
    scaler = MinMaxScaler()
    w2 = w.div(torso_length, axis=0)


    for i, wi in enumerate([w,w2]):
        wi = scaler.fit_transform(wi.values)
        wi =  wi.sum(axis=1)
        wi *= wi
        wi = wi / wi.max()
       # sns.histplot(wi)
   # plt.show()~
    w.div(torso_length, axis=0)
    w = scaler.fit_transform(w.values)
    w =  w.sum(axis=1)
    w *= w
    w = w / w.max()
    


# %%
