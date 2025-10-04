#%%
%run imports_data.py
time = datetime.datetime.now()

%matplotlib inline

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

# Pipeline for regression
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('reduce_dims', PCA()),            
    ('reg', SVR())           
])

# Parameter grid for regression
param_grid = dict(
    reduce_dims__n_components=[4, 6, 8],
    reg__C=np.logspace(-4, 1, 6),
    reg__kernel=['rbf', 'linear']
)

# Grid search
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=1, verbose=2, scoring='r2')
grid.fit(X_train, y_train)

# Test score (R² by default with SVR + scoring='r2')
print("Best params:", grid.best_params_)
print("Best CV R²:", grid.best_score_)
print("Test R²:", grid.score(X_test, y_test))
