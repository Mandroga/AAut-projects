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
import mplcursors


# %% import data
#import data
 
X = np.load('X_train.npy')
y= np.load('y_train.npy')

X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)

df = X_df.copy()
df['target'] = y_df


x_scaler = StandardScaler()
y_scaler = StandardScaler()

df_scaled = df.copy()
feature_names = [col for col in df.columns if col != 'target']
df_scaled[feature_names] = x_scaler.fit_transform(df_scaled[feature_names])
df_scaled['target'] = y_scaler.fit_transform(df_scaled[['target']])
# %% data visualization and information

#info and describe
if 1:
    print(X.shape, y.shape)
    print(df.info())
    print(df.describe)

if 1:
    def plot_f(subplot, n):
        cols = df_scaled.columns
        subplot.plot(df_scaled[cols[n]], df_scaled['target'], label=cols[n], marker='o', linestyle='', markersize=3, alpha=0.7)
        subplot.grid(True)
        subplot.legend()
    min_multiple_plot(6, plot_f)
   # plt.show()

    sns.pairplot(df,hue ='target', palette='viridis')
    plt.show()
    #pairs(df_scaled,target ='target')

print(counts := df['target'].abs().sort_values())

'''
From describe,
There's no missing values, nothing seems strange, mean close to median
From visualization, targets can be distinguished by some of the features with some noise

By visualizing the feature target scatter, there seems to be some relation with the target,
but the feature combination scatters seem to create a more clear distinction between targets!
try multilinear regression with combination of features - create polynomial features see if it becomes linear with target?

Some of the features seem highly correlated, try PCA and lasso for feature reduction, PLS regression

No zeros in target, but small values, 1e-4, smape ?
'''

# %% data histograms
data_for_hist = df_scaled


def plot_f(subplot, n):
    cols = data_for_hist.columns
    sns.histplot(data_for_hist[cols[n]], kde=True, ax=subplot)

min_multiple_plot(7, plot_f)
    
#sns.histplot(data_for_hist, kde=True)
plt.show()
interactive_histogram_plotly(data_for_hist, nbins=50)
plt.show()

'''
Histogram inspection:
0,2 - Seems unifor
1 - Seems normal, but theres a big peak which doesnt follow the distribution
3 - Curved around the edges, uniform in the middle (what is this distribution ??)
4,5 - Looks normal
target- seems skewed normal ?
'''
#%% Feature selection

if 1:
    plot_PCA(df, ['target'])
    plot_PCA(df.drop([3], axis=1), ['target'])
    plot_PCA(df.drop([3,5], axis=1), ['target'])
    plt.show()


df_ = df.copy()
df_.columns = df_.columns.astype(str)
#plot_covmatrix(df_)


stsc = StandardScaler()
X_scaled = stsc.fit_transform(X)
pca = PCA()
pca.fit(X_scaled)
cumulative_variance = pca.explained_variance_ratio_.cumsum()
pca_component_matrix = pd.DataFrame(pca.components_)
print(pca_component_matrix)
print(pca_component_matrix.abs().sum(axis=0)) # Does this give a measure of feature importance ??

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
feature_importance = np.sum(loadings**2, axis=1)
feature_importance /= feature_importance.sum()
print(feature_importance)

'''
PCA plot shows that 2 components explain 0 variance, which matches our observation in the pairplot
where we see that two pairs of features are highly correlated
'''
# %% fitting


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = []
model_names = []

degrees = [0,2,4,6]
regressors = [LinearRegression(), Ridge(), Lasso()]
for degree in degrees:
    for regressor in regressors:
        if degree == 0:
            models.append(Pipeline([('linear', regressor)]))
            model_names.append(f'{regressor.__class__.__name__}')
        else:
            models.append(Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False)),('linear', regressor)]))
            model_names.append(f'{regressor.__class__.__name__} deg {degree}')

print(len(models), len(model_names))

preds = []
score_df = pd.DataFrame(columns=['model','metric','score'])
metrics = [mean_squared_error, mean_absolute_percentage_error, r2_score]
metric_names = ['MSE', 'MAPE', 'R2']

for model, model_name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    preds.append(y_pred)
    for i in range(len(metrics)):
        score_df.loc[len(score_df)] = [model_name, metric_names[i], metrics[i](y_val, y_pred)]

# %% New fitting
preds = {}
score_df = pd.DataFrame(columns=['model','metric','score','train set'])
metrics = [mean_squared_error, mean_absolute_percentage_error, r2_score]
metric_names = ['MSE', 'MAPE', 'R2']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#linear_regressors = [('Linear Regression',LinearRegression()), ('Ridge Regression',Ridge()), ('Lasso',Lasso())]
linear_regressors = [LinearRegression(), Ridge(), Lasso()]
degrees = [1,2,4,6]
models = {'regressor':linear_regressors, 'degrees':degrees}
models_combination = dic_combinator(models)
models = [(Pipeline([('poly', PolynomialFeatures(degree=x['degrees'], include_bias=False)),('linear', x['regressor']))]
print(models_combination)
parameter_grid = {'regressor':linear_regressors, 'degree':degrees,'data':['X']}

parameter_combinations = dic_combinator(validation_grid)

for ps in parameter_combinations:
    regressor_name ,regressor = ps['regressor']
    degree = ps['degree']
    X_train, X_val, y_train, y_val = ps['data']
    
    model = Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False)),('linear', regressor)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    for i in range(len(metrics)):
        score_df.loc[len(score_df)] = [model_name, metric_names[i], metrics[i](y_val, y_pred)]



# %% plot results
def plot_f(subplot, n):
    sns.lineplot(x=y_val.flatten(), y=y_val.flatten(), ax=subplot, color='red')
    sns.scatterplot(x=preds[n], y=y_val, ax=subplot, label=f'MAPE: {scores[n][1]:.2f}\n R2: {scores[n][2]:.2f}')
    subplot.grid()
    subplot.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=10)
    subplot.set_title(model_names[n])


min_multiple_plot(len(models), plot_f, n_cols=3)
#plt.tight_layout()

fig, axes = bar_plot(score_df, y='score', label='metric', min_multiples='model', n_cols=3)


#plt.tight_layout()
plt.show()

'''
As we increase polynomial degree, we see prediction becoming linear, but error on outliers increases
in range 1 to 5, 4 has best scores.


'''






# %%
