# %% data and imports
from turtle import pd


%run imports_data.py
# %% metrics and functions
# ------- Custom winsorized MAPE -------
def winsorized_mape(y_true, y_pred, q=0.95):
    errors = np.abs((y_true - y_pred) / y_true)
    threshold = np.quantile(errors, q)  # cap top q% of errors
    errors = np.clip(errors, 0, threshold)
    return errors.mean()

def score_preds_cv(X, y, kf, score_df, preds, grid):
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

def score_preds_tts(X, y, score_df, preds, grid, test_size=0.2):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
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


metrics = [mean_squared_error, mean_absolute_percentage_error, winsorized_mape, r2_score]
metric_names = ['MSE', 'MAPE', 'wMAPE','R2']


# %% -------------


# %% No feature selection - Models and hyperparameters
# ------- Example regressors & degrees -------
regressors_nofs = [LinearRegression(), Ridge(), Lasso(), PLSRegression(n_components=6)]
degrees_nofs = [1,2,4,6]

param_grid = {'regressor': regressors_nofs,'degree': degrees_nofs}
grid = ParameterGrid(param_grid)

models_nofs = [
    (
        f"{params['regressor'].__class__.__name__} {params['degree']}º",
        Pipeline([
            ('poly', PolynomialFeatures(degree=params['degree'], include_bias=False)),
            ('regressor', params['regressor'])
        ])
    )
    for params in grid
]

grid_nofs = ParameterGrid({'model': models_nofs})
# %% No feature selection - train test split
'''
We want to test different linear regressors,
With different transformation functions (Poly for example)
to see which fits our data best

some regressors have hyperparameters
we will have different transformation functions (Poly different degrees)
we need to find the best combination with validation

Some features might be unnecessary

evaluating with no feature selection
'''
preds_tts_nofs = {}
score_df_tts_nofs = pd.DataFrame(columns=['model','metric','set','score'])
score_df_tts_nofs, preds_tts_nofs = score_preds_tts(X, y, score_df_tts_nofs, preds_tts_nofs, grid_nofs, test_size=0.2)


# %% No feature selection - CV
'''
Fitting with no feature selection
'''
preds_cv_nofs = {}
score_df_cv_nofs = pd.DataFrame(columns=['model','metric','fold','set','score'])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
score_df_cv_nofs, preds_cv_nofs = score_preds_cv(X, y, kf, score_df_cv_nofs, preds_cv_nofs, grid_nofs)


# %% No feature selection - plots
#overfitting comparison

print(overfit_table(score_df_tts_nofs, score_df_cv_nofs, metric_names=['MSE','MAPE','wMAPE','R2'], df_names=['tts','cv']))
# tts
if 0:
    fig, axes = min_multiple_plot(len(models), lambda s, n: plot_f(s,n, (score_df_tts_nofs, preds_tts_nofs)), n_cols=4)
    fig.suptitle('train test split')
# CV 
if 1:
    fig, axes = min_multiple_plot(len(models_nofs), lambda s, n: plot_f(s, n, (score_df_cv_nofs, preds_cv_nofs, grid_nofs)), n_cols=4)
    fig.suptitle('CV')


'''
As we increase polynomial degree, we see prediction becoming linear, but for some points error increases, MAPE explodes
wMAPE, which clips errors above the 95% quantile, shows that validation MAPE is much higher due to the errors above the 95% quantile
Why is CV R2 train-val higher than tts? cv r2 overfitting more than tts ?
MAPE and wMAPE justify cross validation.

'''

# %% --------------


# %% Filter feature selection - Models and hyperparameters

regressors_f = [LinearRegression(), Ridge(), Lasso(), PLSRegression(n_components=6)]
degrees_f = [4,6]

param_grid = {'regressor': regressors_f,'degree': degrees_f}
grid = ParameterGrid(param_grid)

models_f = [
    (
        f"{params['regressor'].__class__.__name__} {params['degree']}º",
        Pipeline([
            ('poly', PolynomialFeatures(degree=params['degree'], include_bias=False)),
            ('drop_corr', DropHighlyCorrelated(threshold=0.99)),
            ('drop_low_target', DropLowTargetCorrelation(threshold=0.01)),
            ('regressor', params['regressor'])
        ])
    )
    for params in grid
]

grid_f = ParameterGrid({'model': models_f})

# %% Filter feature selection - CV
preds_cv_f = {}
score_df_cv_f = pd.DataFrame(columns=['model','metric','fold','set','score'])
kf = KFold(n_splits=2, shuffle=True, random_state=42)
score_df_cv_f, preds_df_cv_f = score_preds_cv(X, y, kf, score_df_cv_f, preds_cv_f, grid_f)

# %% Filter feature selection - plots
# CV 
if 1:
    fig, axes = min_multiple_plot(len(models_f), lambda s, n: plot_f(s, n, (score_df_cv_f, preds_cv_f, grid_f)), n_cols=4)
    fig.suptitle('CV')
#checkboxes
if 0:
    line_by_label_ = get_line_by_label(axes)
    line_by_label = line_by_label_filter(line_by_label_, ['Train','Val'])
    line_by_label['ideal'] = line_by_label_['_child0']
    add_checkbox(line_by_label)


#Mape is still huge, outliers ?

'''

Feature selection is improving scores for Linear Regression and Ridge, but not for Lasso and PLS,
maybe because they can select important features better.

Dominant scores
MAPE - 1.24 Lasso 4º nofs | 1.22 Lasso 6º fs
R2 - 0.89 Lasso 4º nofs | fs
wMAPE - 0.33 Ridge 4º fs | 0.43 Lasso 4º nofs

But we havent tried different hyperparameters yet..

'''
# %% --------------


# %% Wrapper feature selection - Models and hyperparameters
regressors_w = [LinearRegression(), Ridge(), Lasso()]
degrees_w = [1,2,4,6]

regressors = regressors_w
degrees = degrees_w

param_grid = {'regressor': regressors,'degree': degrees}

grid = ParameterGrid(param_grid)

models_w = [
    (
        f"{params['regressor'].__class__.__name__} {params['degree']}º RFE",
        Pipeline([
            ('poly', PolynomialFeatures(degree=params['degree'], include_bias=False)),
            ('RFE', RFECV(estimator=params['regressor'], step=0.1, cv=5, scoring='neg_mean_squared_error')),
            ('regressor', params['regressor'])
        ])
    )
    for params in grid
]

grid_w = ParameterGrid({'model': models_w})


# %% Wrapper feature selection - CV
preds_cv_w = {}
score_df_cv_w = pd.DataFrame(columns=['model','metric','fold','set','score'])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
score_df_cv_w, preds_cv_w = score_preds_cv(X, y, kf, score_df_cv_w, preds_cv_w, grid_w)
# %% Wrapper feature selection - plots
# CV - RFE
if 1:
    fig, axes = min_multiple_plot(len(models_w), lambda s, n: plot_f(s, n, (score_df_cv_w, preds_cv_w, grid_w)), n_cols=None)
    fig.suptitle('CV with RFE')
#checkboxes
if 0:
    line_by_label_ = get_line_by_label(axes)
    line_by_label = line_by_label_filter(line_by_label_, ['Train','Val'])
    line_by_label['ideal'] = line_by_label_['_child0']
    add_checkbox(line_by_label)

    bar_plot_df = score_df_cv_f[score_df_cv_f['metric'] != 'MSE']

    fig, axes = bar_plot(bar_plot_df[bar_plot_df['set']=='val_p'], y='score', label='metric', min_multiples='model', n_cols=2)
    fig.suptitle('scores on cv with feature selection')
#heatmap cv
if 0:
    heat_map_df_f = score_df_cv_f.query('set == "val_p" and metric == "wMAPE"')
    heat_map_df_f['model name'] = heat_map_df_f['model'].str.split(' ').str[0]
    heat_map_df_f['degree'] = heat_map_df_f['model'].str.split(' ').str[1].str.replace('º','')
    heat_map_df_f.drop(['metric','model','set'],axis=1,inplace=True)
    heat_map_df_f = heat_map_df_f.pivot_table(index='model name', columns='degree', values='score')
    print(heat_map_df_f)

    fig, ax = plt.subplots()
    ax.set_title('wMAPE score heatmap with feature selection')
    sns.heatmap(heat_map_df_f, annot=True, cmap='coolwarm', ax=ax)


# %% --------------
# %% show plots
plt.show()
# %% clear plots
plt.close('all')
# %%
