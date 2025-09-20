# %% data and imports
%run imports_data.py
# %% Fitting - Models and hyperparameters
# ------- Custom winsorized MAPE -------
def winsorized_mape(y_true, y_pred, q=0.95):
    errors = np.abs((y_true - y_pred) / y_true)
    threshold = np.quantile(errors, q)  # cap top q% of errors
    errors = np.clip(errors, 0, threshold)
    return errors.mean()

# ------- Metrics -------
metrics = [mean_squared_error, mean_absolute_percentage_error, winsorized_mape, r2_score]
metric_names = ['MSE', 'MAPE', 'wMAPE','R2']

# ------- Example regressors & degrees -------
regressors_nofs = [LinearRegression(), Ridge(), Lasso(), PLSRegression(n_components=6)]
degrees_nofs = [1,2,4,6]

regressors = regressors_nofs
degrees = degrees_nofs

param_grid = {'regressor': regressors,'degree': degrees}

grid = ParameterGrid(param_grid)

models = [
    (
        f"{params['regressor'].__class__.__name__} {params['degree']}º",
        Pipeline([
            ('poly', PolynomialFeatures(degree=params['degree'], include_bias=False)),
            ('regressor', params['regressor'])
        ])
    )
    for params in grid
]

grid = ParameterGrid({'model': models})
# %% Fitting - train test split
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

preds_tts = {}
score_df_tts = pd.DataFrame(columns=['model','metric','set','score'])
preds = {}
score_df = pd.DataFrame(columns=['model','metric','set','score'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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

score_df_tts = score_df
preds_tts = preds
print(score_df.columns)
print(score_df.shape)

# %% Fitting - CV
'''
Fitting with no feature selection
'''

# ------- Prepare results -------
preds_cv = {}
score_df_cv = pd.DataFrame(columns=['model','metric','fold','set','score'])

# ------- K-Fold CV -------
kf = KFold(n_splits=2, shuffle=True, random_state=42)

X = X.values if hasattr(X,'values') else X  # convert to array if dataframe
y = y.values if hasattr(y,'values') else y

for ps in grid:
    model_name, model = ps['model']

    iter_name = model_name
    preds_cv[iter_name] = {'train_p':[],'train_t':[],'val_p':[],'val_t':[]}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Fit and predict
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        preds_cv[iter_name]['train_p'].append(y_pred_train)
        preds_cv[iter_name]['val_p'].append(y_pred_val)
        preds_cv[iter_name]['train_t'].append(y_train)
        preds_cv[iter_name]['val_t'].append(y_val)
        
        # Compute metrics
        sets = [['train_p', y_train, y_pred_train], ['val_p', y_val, y_pred_val]]
        for set_name, truth, pred in sets:
            for i, metric in enumerate(metrics):
                score_df_cv.loc[len(score_df_cv)] = [
                    model_name, metric_names[i], fold+1, set_name, metric(truth, pred)
                ]

# ------- Summary: mean ± std per metric per model -------
score_df_cv = score_df_cv.groupby(['model','metric','set']).score.agg(['mean']).reset_index()
score_df_cv['score']=score_df_cv['mean']
score_df_cv.drop('mean',axis=1,inplace=True)

# %% plot results
def plot_f(subplot, n, data):
    score_df_, preds_ = data
    model_name, _ = grid[n]['model']
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
    y_set = {'train_p':y_train, 'val_p':y_val}
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

# tts
if 0:
    fig, axes = min_multiple_plot(len(models), lambda s, n: plot_f(s,n, (score_df, preds)), n_cols=4)
    fig.suptitle('train test split')
# CV 
if 0:
    fig, axes = min_multiple_plot(len(models), lambda s, n: plot_f(s, n, (score_df_cv, preds_cv)), n_cols=4)
    fig.suptitle('CV')
#checkboxes
if 0:
    line_by_label_ = get_line_by_label(axes)
    line_by_label = line_by_label_filter(line_by_label_, ['Train','Val'])
    line_by_label['ideal'] = line_by_label_['_child0']
    add_checkbox(line_by_label)

# bar plot cv
if 1:
    bar_plot_df = score_df_cv[score_df_cv['metric'] != 'MSE']

    fig, axes = bar_plot(bar_plot_df[bar_plot_df['set']=='val_p'], y='score', label='metric', min_multiples='model', n_cols=4)
    fig.suptitle('scores on cv no feature selection')
#heatmap cv
if 0:
    heat_map_df = score_df_cv.query('set == "val_p" and metric == "wMAPE"')
    heat_map_df['model name'] = heat_map_df['model'].str.split(' ').str[0]
    heat_map_df['degree'] = heat_map_df['model'].str.split(' ').str[1].str.replace('º','')
    heat_map_df.drop(['metric','model','set'],axis=1,inplace=True)
    heat_map_df = heat_map_df.pivot_table(index='model name', columns='degree', values='score')
    print(heat_map_df)
    
    fig, ax = plt.subplots()
    ax.set_title('wMAPE score heatmap')
    sns.heatmap(heat_map_df, annot=True, cmap='coolwarm', ax=ax)

#plt.show()

'''
As we increase polynomial degree, we see prediction becoming linear, but for some points error increases, MAPE explodes
wMAPE, which clips errors above the 95% quantile, shows that validation MAPE is much higher due to the errors above the 95% quantile


'''

# %% Fitting with feature selection

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


regressors_fs = [LinearRegression(), Ridge(), Lasso(), PLSRegression(n_components=6)]
degrees_fs = [4,6]

regressors = regressors_fs
degrees= degrees_fs

param_grid = {'regressor': regressors,'degree': degrees}

grid = ParameterGrid(param_grid)

models = [
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

grid = ParameterGrid({'model': models})


# ------- Prepare results -------
preds_cv_fs = {}
score_df_cv_fs = pd.DataFrame(columns=['model','metric','fold','set','score'])

preds = preds_cv_fs
score_df = score_df_cv_fs

# ------- K-Fold CV -------
kf = KFold(n_splits=2, shuffle=True, random_state=42)

X = X.values if hasattr(X,'values') else X  # convert to array if dataframe
y = y.values if hasattr(y,'values') else y

for ps in grid:
    model_name, model = ps['model']

    iter_name = model_name
    preds[iter_name] = {'train_p':[],'train_t':[],'val_p':[],'val_t':[]}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Fit and predict
        model.fit(X_train, y_train)

        if 1:
            print(model_name)
            X_poly = model.named_steps['poly'].transform(X_train)
            print("Features after polynomial expansion:", X_poly.shape[1])

            # After dropping highly correlated features
            X_drop_corr = model.named_steps['drop_corr'].transform(X_poly)
            print("Features after dropping correlated features:", X_drop_corr.shape[1])

            # After dropping low correlation with target
            X_final = model.named_steps['drop_low_target'].transform(X_drop_corr)
            print("Features after dropping low target correlation:", X_final.shape[1])

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

preds_cv_fs = preds
score_df_cv_fs = score_df

# %% plot results - feature selection
def plot_f(subplot, n, data):
    score_df_, preds_ = data
    model_name, _ = grid[n]['model']
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
    y_set = {'train_p':y_train, 'val_p':y_val}
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

# tts
if 0:
    fig, axes = min_multiple_plot(len(models), lambda s, n: plot_f(s,n, (score_df, preds)), n_cols=4)
    fig.suptitle('train test split')
# CV 
if 0:
    fig, axes = min_multiple_plot(len(models), lambda s, n: plot_f(s, n, (score_df_cv_fs, preds_cv_fs)), n_cols=4)
    fig.suptitle('CV')
#checkboxes
if 0:
    line_by_label_ = get_line_by_label(axes)
    line_by_label = line_by_label_filter(line_by_label_, ['Train','Val'])
    line_by_label['ideal'] = line_by_label_['_child0']
    add_checkbox(line_by_label)

# bar plot cv fs
if 1:
    bar_plot_df = score_df_cv_fs[score_df_cv_fs['metric'] != 'MSE']

    fig, axes = bar_plot(bar_plot_df[bar_plot_df['set']=='val_p'], y='score', label='metric', min_multiples='model', n_cols=2)
    fig.suptitle('scores on cv with feature selection')
#heatmap cv
if 0:
    heat_map_df_fs = score_df_cv_fs.query('set == "val_p" and metric == "wMAPE"')
    heat_map_df_fs['model name'] = heat_map_df_fs['model'].str.split(' ').str[0]
    heat_map_df_fs['degree'] = heat_map_df_fs['model'].str.split(' ').str[1].str.replace('º','')
    heat_map_df_fs.drop(['metric','model','set'],axis=1,inplace=True)
    heat_map_df_fs = heat_map_df_fs.pivot_table(index='model name', columns='degree', values='score')
    print(heat_map_df_fs)

    fig, ax = plt.subplots()
    ax.set_title('wMAPE score heatmap with feature selection')
    sns.heatmap(heat_map_df_fs, annot=True, cmap='coolwarm', ax=ax)

plt.show()

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
# %%
