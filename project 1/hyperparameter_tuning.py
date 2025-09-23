# %% data and imports
from turtle import pd


%run imports_data.py
# %% metrics
# ------- Custom winsorized MAPE -------
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
    fig.suptitle('CV - No Feature Selection')


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
    fig.suptitle('CV - Filter Feature Selection')
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
    fig, axes = min_multiple_plot(len(models_w), lambda s, n: plot_f(s, n, (score_df_cv_w, preds_cv_w, grid_w)), n_cols=3)
    fig.suptitle('CV - RFE Feature Selection')
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

# %% Final training
ols = LinearRegression()
ridge = Ridge()
lasso = Lasso(max_iter=5000)
pls = PLSRegression()


pipe = Pipeline([
    ("poly", PolynomialFeatures()),
    ("scaler", StandardScaler()),
    ("selector", "passthrough"),
    ("regressor", "passthrough"),
])

degrees = Integer(1,6)
rfe_cv_folds = 2
pls_components = Integer(1,10)
alpha_space = Real(1e-3, 1e3, prior="log-uniform")

combined_selector = Pipeline([
    ('high_corr', DropHighlyCorrelated(threshold=0.95)),
    ("corr_filter", DropLowTargetCorrelation(threshold=0.01)),
    ("rfe", RFECV(LinearRegression(), cv=KFold(5), scoring="neg_mean_squared_error"))
])

selectors = [
    "passthrough",  # no feature selection
    RFECV(LinearRegression(), cv=KFold(rfe_cv_folds), scoring="neg_mean_squared_error"),
    combined_selector
]

search_spaces = [
    {
        "poly__degree": degrees,
        "selector": selectors,
        "regressor": [ols],
    }
]
search_spaces += [
    {
        "poly__degree": degrees,
        "selector": selectors,
        "regressor": [model],
        "regressor__alpha": alpha_space,
    }
    for model in [ridge, lasso]
]
search_spaces += [
    {
        "poly__degree": degrees,
        "selector": ["passthrough"],  # keep all features
        "regressor": [PLSRegression()],
        "regressor__n_components": pls_components,
    },
]

# Bayesian search
opt = BayesSearchCV(
    estimator=pipe,
    search_spaces=search_spaces,
    n_iter=5,  # number of trials (increase for better search)
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

opt.fit(X, y)
cv_results = pd.DataFrame(opt.cv_results_)
# %% show plots
plt.show()
# %% clear plots
plt.close('all')
# %%
