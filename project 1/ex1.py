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
# %% data visualization and information

#info and describe
if 1:
    print(X.shape, y.shape)
    print(df.info())
    print(df.describe)

#data pairplot and Xi,Y scatter
if 1:
    def plot_f(subplot, n):
        cols = df_scaled.columns
        subplot.plot(df_scaled[cols[n]], df_scaled['target'], label=cols[n], marker='o', linestyle='', markersize=3, alpha=0.7)
        subplot.grid(True)
        subplot.legend()
    min_multiple_plot(6, plot_f)
    sns.pairplot(df,hue ='target', palette='viridis')
    plt.show()

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
#%% Feature analysis

#PCA drop original features
if 0:
    plot_PCA(df, ['target'])
    plot_PCA(df.drop([3], axis=1), ['target'])
    plot_PCA(df.drop([3,5], axis=1), ['target'])
    plt.show()
#covmatrix
if 0:
    df_ = df.copy()
    df_.columns = df_.columns.astype(str)
    plot_covmatrix(df_)


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

#Testing Poly Features
if 0:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_df = pd.DataFrame(X_train)
    X_train_df.columns = X_train_df.columns.astype(str)
    y_train_df = pd.DataFrame(y_train)
    degrees = [1, 2, 4, 6]
    all_corrs = []

    y_series = y_train_df.iloc[:, 0]  # target as Series

    for d in degrees:
        poly = PolynomialFeatures(degree=d, include_bias=False)
        X_poly = pd.DataFrame(poly.fit_transform(X_train_df))
        feature_names = poly.get_feature_names_out(X_train_df.columns)
        X_poly.columns = feature_names
        
        # Absolute correlations with target
        corrs = X_poly.apply(lambda x: x.corr(y_series)).values
        
        # Add degree info for plotting
        df_plot = pd.DataFrame({'Degree': [d]*len(corrs), 'Features':feature_names, 'Correlation': corrs})
        all_corrs.append(df_plot)

    # Combine all degrees
    plot_df = pd.concat(all_corrs, ignore_index=True)
    plot_df_abs = plot_df.copy()
    plot_df_abs = plot_df_abs.drop_duplicates(subset='Features', keep='first')
    plot_df_abs['Correlation'] = plot_df_abs['Correlation'].abs()

    plot_df_abs['feature_set'] = plot_df_abs['Features'].apply(lambda x: set(str(x).split()))

    # Keep only dominant feature sets
    dominant_rows = []

    for i, row_i in plot_df_abs.iterrows():
        is_dominated = False
        for j, row_j in plot_df_abs.iterrows():
            if i == j:
                continue
            # row_j set is a subset of row_i and has higher correlation
            if row_j['feature_set'].issubset(row_i['feature_set']) and row_j['Correlation'] > row_i['Correlation']:
                is_dominated = True
                break
        if not is_dominated:
            dominant_rows.append(i)

    plot_df_abs_dominant = plot_df_abs.loc[dominant_rows].reset_index(drop=True)
    pd.set_option('display.max_rows', None)      # None = no limit
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)    
    print(plot_df_abs_dominant.sort_values(by='Correlation', ascending=False))

    # Violin plot
    #sns.violinplot(x='Degree', y='Correlation', data=plot_df, inner="point")
   # plt.ylabel("Absolute Correlation with Target")
   # plt.show()
            
        



    #plot
    if 0:
        df = X_poly
        df['target'] = y_df

        def plot_f(subplot, n):
            cols = X_poly.columns
            subplot.plot(X_poly[cols[n]], X_poly['target'], label=cols[n], marker='o', linestyle='', markersize=3, alpha=0.7)
            subplot.grid(True)
            subplot.legend()
        #min_multiple_plot(len(X_poly.columns), plot_f)
        #sns.pairplot(df,hue ='target', palette='viridis')
        col_comb = list(itertools.combinations(df.drop('target',axis=1).columns, 2))
        print(col_comb)
        def plot_f2(subplot, n):
            c1,c2 = col_comb[n]
            sns.scatterplot(data=df,x=c1, y=c2,hue="target",palette="viridis",s=80,ax=subplot, legend=False)
            #subplot.set_title(f"x={c1}, y={c2}")
        min_multiple_plot(len(col_comb), plot_f2)
        plt.show()
'''
PCA plot shows that 2 components explain 0 variance, which matches our observation in the pairplot
where we see that two pairs of features are highly correlated

Beware! Features might not be correlated after Transformation function!

Visualizing Poly Features ---
2nd degree - squared features are not seperating very well, but some feature products sums are becoming linear with target
(x3*x4+x3*x5) and (x2*x4+x2*x5)

'''

# %% Feature selection
'''
Three types of methods: Filter, embedded and wrapper 
We will use filter -  and wrapper
'''

#Filter methods - Corr with target, Var threshold, MI with target
if 1:
    x_scaler = StandardScaler()
    X_scaled = pd.DataFrame(x_scaler.fit_transform(X))
    X_scaled.columns = X_scaled.columns.astype(str)
    poly = PolynomialFeatures(degree=6, include_bias=False)
    X_scaled_poly = pd.DataFrame(poly.fit_transform(X_scaled))
    feature_names = poly.get_feature_names_out(X_scaled.columns)
    X_scaled_poly.columns = feature_names
    
    var_thresh = VarianceThreshold(threshold=1e-3)
    X_scaled_poly = pd.DataFrame(var_thresh.fit_transform(X_scaled_poly), columns=X_scaled_poly.columns[var_thresh.get_support()])
    
    #MI -----
    mi = mutual_info_regression(X_df, y_df.iloc[:,0])
    mi_series = pd.Series(mi, index=X_df.columns).sort_values(ascending=False)
    print("Mutual Information (top features):")
    print(mi_series.head(6))

#RFE feature selection
if 0:
    degrees = [1,2,4,6]
    linear_regressors = [LinearRegression()]
    for d in degrees:
        model = linear_regressors[0]
        
        poly = PolynomialFeatures(degree=d, include_bias=False)
        X_poly = pd.DataFrame(poly.fit_transform(X_df))
        feature_names = poly.get_feature_names_out(X_df.columns)
        X_poly.columns = feature_names

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        selector = RFECV(model, cv=cv, scoring='r2') 
        selector = selector.fit(X_poly, y)

        print("Optimal number of features:", selector.n_features_)
        print("Selected features:", list(X_poly.columns[selector.support_]))

'''

'''


 # %% Fitting
'''
We want to test different linear regressors,
With different transformation functions (Poly for example)
to see which fits our data best

some regressors have hyperparameters
we will have different transformation functions (Poly different degrees)
we need to find the best combination with validation

Some features might be unnecessary
'''


preds = {}
score_df = pd.DataFrame(columns=['model','metric','set','score'])
metrics = [mean_squared_error, mean_absolute_percentage_error, r2_score]
metric_names = ['MSE', 'MAPE', 'R2']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

regressors = [LinearRegression(), Ridge(), Lasso(), PLSRegression()]
degrees = [1,2,4,6]


#list or dict combinator
if 0:
    models = [degrees, linear_regressors]
    model_combinations = [list(p) for p in itertools.product(*models)]
    models = [(f'{regressor.__class__.__name__} {degree}º',Pipeline([('poly', PolynomialFeatures(degree=degree, include_bias=False)),('regressor', regressor)])) for regressor, degree in model_combinations]
else:
    models = {'degree':degrees, 'regressor':linear_regressors}
    model_combinations = dic_combinator(models)
    models = [(f'{comb['regressor'].__class__.__name__} {comb['degree']}º',Pipeline([('poly', PolynomialFeatures(degree=comb['degree'], include_bias=False)),('regressor', comb['regressor'])])) for comb in model_combinations]

parameter_grid = {'model':models}
parameter_combinations = dic_combinator(parameter_grid)

for ps in parameter_combinations:
    model_name, model = ps['model']
    
    iter_name = model_name
    
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    preds[iter_name] = {}
    preds[iter_name]['train'] = y_pred_train
    preds[iter_name]['val'] = y_pred_val
    
    sets = [['train',y_train, y_pred_train], ['val', y_val, y_pred_val]]
    for set_name, truth, pred in sets:
        for i in range(len(metrics)):
            score_df.loc[len(score_df)] = [model_name, metric_names[i],set_name, metrics[i](truth, pred)]



# %% plot results

def plot_f(subplot, n):
    model_name, _ = parameter_combinations[n]['model']
    iter_name = model_name
    scores = []
    for set_name in ['train','val']:
        for metric in ['MAPE', 'R2']:
            scores.append(score_df.query(f'model == "{model_name}" and set == "{set_name}" and metric == "{metric}"')['score'].iloc[0])
    #MAPE_train = score_df.query(f'model == "{model_name}" and set == "train" and metric == "MAPE"')['score'].iloc[0]
    #R2_train = score_df.query(f'model == "{model_name}" and metric == "R2"')['score'].iloc[0]
    sns.lineplot(x=y_val.flatten(), y=y_val.flatten(), ax=subplot, color='red')
    sns.scatterplot(x=preds[iter_name]['train'], y=y_train, ax=subplot, label=f'Train\nMAPE: {scores[0]:.2f}\n R2: {scores[1]:.2f}',alpha=0.5)
    sns.scatterplot(x=preds[iter_name]['val'], y=y_val, ax=subplot, label=f'Validation\nMAPE: {scores[2]:.2f}\n R2: {scores[3]:.2f}')
    subplot.grid()
    subplot.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize=10)
    subplot.set_title(model_name)

fig, axes = bar_plot(score_df[score_df['set']=='val'], y='score', label='metric', min_multiples='model', n_cols=4)

fig, axes = min_multiple_plot(len(models), plot_f, n_cols=4)

#checkboxes
if 1:
    line_by_label_ = get_line_by_label(axes)
    line_by_label = line_by_label_filter(line_by_label_, ['Train','Val'])
    line_by_label['ideal'] = line_by_label_['_child0']
    add_checkbox(line_by_label)
else:
    plt.show()


'''
As we increase polynomial degree, we see prediction becoming linear, but for some points error increases

'''






# %%
