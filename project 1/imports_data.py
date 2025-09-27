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
from skopt.space import Real, Integer, Categorical
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import trim_mean
from sklearn.base import BaseEstimator, TransformerMixin, clone
import joblib
from sklearn.decomposition import PCA
from xgboost import XGBRegressor



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

RANDOM_STATE = 42


# %% Useful functions

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

def plot_f(subplot, n, data):
    score_df_, preds_, grid_ = data
    model_name, _ = grid_[n]['model']
    iter_name = model_name

    #scores
    scores = {}
    set_names = ['train_p','val_p']
    used_metric_names = ['MAPE', 'wMAPE', 'R2']
    for set_name in set_names:
        scores[set_name] = {}
        for metric in used_metric_names:
            scores[set_name][metric] = score_df_.query(f'model == "{model_name}" and set == "{set_name}" and metric == "{metric}"')['score'].iloc[0]

    #ideal

    target = preds_[iter_name]['val_t'] 
    y = np.concatenate(target)
    sns.lineplot(x=y, y=y, ax=subplot, color='red')
    
    #scatters
    alphas = {'train_p':0.5,'val_p':1}
    for set_name in set_names:
         txt = set_name
         for metric in used_metric_names:
             txt += f'\n{metric}: {scores[set_name][metric]:.2f}'
         x = preds_[iter_name][set_name]
         target = preds_[iter_name][set_name.replace('_p','_t')] 
         x = np.concatenate(x)
         y = np.concatenate(target)
         sns.scatterplot(x=x, y=y, ax=subplot, label=txt,alpha=alphas[set_name])

  
    subplot.grid()
    subplot.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=10)
    subplot.set_title(model_name)


class DropHighlyCorrelated(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        corr_matrix = pd.DataFrame(X).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]
        return self

    def transform(self, X):
        return pd.DataFrame(X).drop(columns=self.to_drop_, errors='ignore').values

class DropLowTargetCorrelation(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.features_to_keep_ = None

    def fit(self, X, y):
        df = pd.DataFrame(X)
        corr = df.corrwith(pd.Series(y)).abs()
        self.features_to_keep_ = corr[corr > self.threshold].index
        return self

    def transform(self, X):
        return pd.DataFrame(X).iloc[:, self.features_to_keep_].values

# %% metrics
# ------- Custom winsorized MAPE -------
metrics = [mean_squared_error, mean_absolute_percentage_error, winsorized_mape, r2_score]
metric_names = ['MSE', 'MAPE', 'wMAPE','R2']


# %%
