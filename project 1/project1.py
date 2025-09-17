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
if 0:
    print(X.shape, y.shape)
    print(df.info())
    print(df.describe)

if 0:
    def plot_f(subplot, n):
        cols = df_scaled.columns
        subplot.plot(df_scaled[cols[n]], df_scaled['target'], label=cols[n], marker='o', linestyle='', markersize=3, alpha=0.7)
        subplot.grid(True)
        subplot.legend()
    min_multiple_plot(6, plot_f)
    plt.show()

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
    
plt.show()

#sns.histplot(data_for_hist, kde=True)
#plt.show()
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

# %% fitting

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = []
model_names = []

degrees = [0,2,4]
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
scores = []
score_df = pd.DataFrame(columns=['model','metric','score'])
metrics = [mean_squared_error, mean_absolute_percentage_error, r2_score]
metric_names = ['MSE', 'MAPE', 'R2']
for model, model_name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    preds.append(y_pred)
    scores.append((mean_squared_error(y_val, y_pred), mean_absolute_percentage_error(y_val, y_pred), r2_score(y_val, y_pred)))
    for i in range(len(metrics)):
        score_df.loc[len(score_df)] = [model_name, metric_names[i], scores[-1][i]]

# %% plot results
def plot_f(subplot, n):
    sns.lineplot(x=y_val.flatten(), y=y_val.flatten(), ax=subplot, color='red')
    sns.scatterplot(x=preds[n], y=y_val, ax=subplot, label=f'MAPE: {scores[n][1]:.2f}\n R2: {scores[n][2]:.2f}')
    subplot.grid()
    subplot.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=5)
    subplot.set_title(model_names[n])

if 0:
    min_multiple_plot(len(models), plot_f)
    plt.tight_layout()
    plt.show()

fig, axes = bar_plot(score_df, y='score', label='metric', min_multiples='model')


def on_click(event):
    # Check which Axes was clicked
    axes_flatten = list(axes.flatten())
    ax_index = axes_flatten.index(event.inaxes)
    #n_rows, n_cols = min_multiple_plot_format(len(models))
   # i, j = ax_index // n_cols, ax_index % n_cols
    model_name = model_names[ax_index]
    df_ = score_df[(score_df['model']==model_name)]
    fig2, axes2 = bar_plot(df_, y='score', label='metric', min_multiples='model')
    plt.show()


#fig.canvas.mpl_connect('button_press_event', on_click)

plt.tight_layout()
plt.show()

'''
As we increase polynomial degree, we see prediction becoming linear, but error on outliers increases
in range 1 to 5, 4 has best scores.


'''




# %%
