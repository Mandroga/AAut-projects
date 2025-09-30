#%%
%run imports_data.py

time = datetime.datetime.now()

#%%
%matplotlib inline

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

#%%  

# Dataset
X = np.load("X_train.npy")
Y= np.load('y_train.npy')

#%%
# Plot
for i in range(6):
    plt.figure()
    plt.scatter(X[:,i], Y, c='k', marker='o')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Feature {i} vs Target")
plt.show()

#%%

def smape(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)   
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def biasssmape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    Works with negative and positive values
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_pred - y_true)/np.sum((np.abs(y_true) + np.abs(y_pred)))    

# %%                            
class RBFFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=1.0, M=5):
        self.sigma = sigma
        self.M = M
        self.centroids = None

    def fit(self, X, y=None):
        kmeans = KMeans(n_clusters=self.M, random_state=42)
        kmeans.fit(np.array(X))
        self.centroids = kmeans.cluster_centers_  # shape: (M, N_features)
        if self.sigma is None:
            # Computing the spread (sigma) as the average distance between centroids
            if self.M > 1:
                dists = np.linalg.norm(self.centroids[:, np.newaxis] - self.centroids, axis=2)
                self.sigma = np.mean(dists[dists != 0])  # Avoid zero distance to itself
            else:
                self.sigma = 1.0  # Arbitrary value when there's only one centroid
        return self

    def transform(self, X):
        N = X.shape[0]
        M = self.centroids.shape[0]
        Phi = np.zeros((N, M))
        diffs = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        Phi = np.exp(-np.sum(diffs**2, axis=2) / (2 * self.sigma**2))
        return Phi

 #%% Hyperparameter Tuning and Model Evaluation
cv=3
factor= (cv-1)/cv
min_val = min(int(np.sqrt(X.shape[0]*factor)), int(X.shape[0]*factor))
max_val = max(int(np.sqrt(X.shape[0]*factor)), int(X.shape[0]*factor))
rfe_cv_folds = 5
print(min_val, max_val)

pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("RBS", RBFFeatures(M=10)),      # placeholder M (overridden)
    ("reg", LinearRegression())      # placeholder model (overridden)
])

pipefs = Pipeline([
    ("scaler", RobustScaler()),
    ("RBS", RBFFeatures(M=10)), 
    ('drop_corr', DropHighlyCorrelated(threshold=0.99)),
    ('drop_low_target', DropLowTargetCorrelation(threshold=0.01)),     # placeholder M (will be overridden)
    ("reg", LinearRegression())      # placeholder model (overridden)
])

pipepca = Pipeline([    

    ('scale', RobustScaler()),  
    ('RBS', RBFFeatures(M=10)),
    ('reduce_dims', PCA()),            
    ('reg', LinearRegression())           
])


pipe_pca_rfe = Pipeline([
    ('scale', RobustScaler()),
    ("RBS", RBFFeatures(M=10)),
    ("rfe", RFE(estimator=LinearRegression())),  # choose desired number
    ('reduce_dims', PCA(n_components=0.99)),
    ("reg", LinearRegression())
])


"""
pipe_fs_rfe = Pipeline([
    ("scaler", RobustScaler()),
  
    ("RBS", RBFFeatures(M=10)), 
    ('drop_corr', DropHighlyCorrelated(threshold=0.99)),
    ('drop_low_target', DropLowTargetCorrelation(threshold=0.01)),     # placeholder M (will be overridden)
    ("reg", LinearRegression())      # placeholder model (overridden)
])
"""
#mx_val = min(max_val, 300)  # limit max M to 30 for computational reasons

param_grid = [
    # LinearRegression: no alpha
    {
        "RBS__M": range(min_val, max_val),   # or range(min_val, max_val+1)
        "reg": [LinearRegression()],
        "reg__fit_intercept": [True, False],
    },
    # Ridge: tune alpha
    {
        "RBS__M": range(min_val, max_val),
        "reg": [Ridge(max_iter=5000)],
        "reg__alpha": [0.001,0.005,0.01],
    },
    #PLS
    {
        "RBS__M": range(min_val, max_val),
        "reg": [PLSRegression(max_iter=5000)],
        "reg__n_components": range(2, min(10, X.shape[1]))
    },
    # SVR
    {
        "RBS__M": range(min_val, max_val),
        "reg": [SVR()],
        "reg__C": [10.0,12.0,14.0],
        "reg__kernel": ['rbf', 'linear']
    }
]

# For PCA

param_grid_pca = [
    # LinearRegression
    {
        "RBS__M": range(min_val, max_val),
        "reg": [LinearRegression()],
        "reg__fit_intercept": [True, False],
        "reduce_dims__n_components": [5, 15, 0.95]  
    },
    # Ridge
    {
        "RBS__M": range(min_val, max_val),
        "reg": [Ridge(max_iter=5000)],
        "reg__alpha": [0.01, 0.1, 1.0],
        "reduce_dims__n_components": [5, 15, 0.95]
    },
    # PLS
    {
        "RBS__M": range(min_val, max_val),
        "reg": [PLSRegression(max_iter=5000)],
        "reg__n_components": range(2, min(10, X.shape[1])),
        "reduce_dims__n_components": [5, 15, 0.95]
    },
    # SVR
    {
        "RBS__M": range(min_val, max_val),
        "reg": [SVR()],
        "reg__C": [0.1, 1.0, 10.0],
        "reg__kernel": ['rbf', 'linear'],
        "reduce_dims__n_components": [5, 15, 0.95]
    }
]

# For pca + rfe
param_grid_pca_rfe = [
    # LinearRegression
    {
        "RBS__M": range(min_val, max_val),
        "reg": [LinearRegression()],
        "reg__fit_intercept": [True, False],
        "reduce_dims__n_components": [5, 15, 0.95],
        'rfe__n_features_to_select': [3, 5, 7, 10],
    },
    # Ridge
    {
        "RBS__M": range(min_val, max_val),
        "reg": [Ridge(max_iter=5000)],
        "reg__alpha": [0.01, 0.1, 1.0],
        "reduce_dims__n_components": [5, 15, 0.95],
        "rfe__n_features_to_select": [5, 10, 15]
    },
    # SVR
    {
        "RBS__M": range(min_val, max_val),
        "reg": [SVR()],
        "reg__C": [0.1, 1.0, 10.0],
        "reg__kernel": ['rbf', 'linear'],
        "reduce_dims__n_components": [5, 15, 0.95],
        "rfe__n_features_to_select": [5, 10, 15]
    }
]


# For Bayes Search
param_space = [
    # LinearRegression
    {
        "RBS__M": Integer(min_val, max_val),  
        "reg": Categorical([LinearRegression()]),
        "reg__fit_intercept": Categorical([True]),
    },
    # Ridge: tune alpha
    {
        "RBS__M": Integer(min_val, max_val),
        "reg": Categorical([Ridge(max_iter=5000)]),
        "reg__alpha": Real(0.01, 1.0, prior="log-uniform"),  
    },
    # Lasso: tune alpha
    {
        "RBS__M": Integer(min_val, max_val),
        "reg": Categorical([Lasso(max_iter=10000)]),
        "reg__alpha": Real(0.0001, 0.001, prior="log-uniform"),
    },
    # PLS
    {
    "RBS__M": Integer(min_val, max_val),
    "reg": Categorical([PLSRegression(max_iter=5000)]),
    "reg__n_components": Integer(2, min(10, X.shape[1]) - 1)
    },
    # SVR
    {
    "RBS__M": Integer(min_val, max_val),
    "reg": Categorical([SVR()]),
    "reg__C": Real(0.1, 10.0, prior="log-uniform"),  # continuous search
    "reg__kernel": Categorical(['rbf', 'linear'])
}
]

from skopt.space import Integer, Real, Categorical

param_space_pca = [
    # LinearRegression
    {
        "RBS__M": Integer(min_val, max_val),
        "reg": Categorical([LinearRegression()]),
        "reg__fit_intercept": Categorical([True, False]),
        "reduce_dims__n_components": Categorical([5, 15, 0.95])
    },
    # Ridge
    {
        "RBS__M": Integer(min_val, max_val),
        "reg": Categorical([Ridge(max_iter=5000)]),
        "reg__alpha": Real(0.01, 1.0, prior="log-uniform"),
        "reduce_dims__n_components": Categorical([5, 15, 0.95])
    },
    # PLS
    {
        "RBS__M": Integer(min_val, max_val),
        "reg": Categorical([PLSRegression(max_iter=5000)]),
        "reg__n_components": Integer(2, min(10, X.shape[1])-1),
        "reduce_dims__n_components": Categorical([5, 15, 0.95])
    },
    # SVR
    {
        "RBS__M": Integer(min_val, max_val),
        "reg": Categorical([SVR()]),
        "reg__C": Real(0.1, 10.0, prior="log-uniform"),
        "reg__kernel": Categorical(['rbf', 'linear']),
        "reduce_dims__n_components": Categorical([5, 15, 0.95])
    }
]

param_space_pca_rfe = [
    # LinearRegression
    {
        "RBS__M": Integer(min_val, max_val),
        "reg": Categorical([LinearRegression()]),
        "reg__fit_intercept": Categorical([True, False]),
        "reduce_dims__n_components": Categorical([2, 0.95]),
        "rfe__n_features_to_select": Integer(5, 15)
    },
    # Ridge
    {
        "RBS__M": Integer(min_val, max_val),
        "reg": Categorical([Ridge(max_iter=5000)]),
        "reg__alpha": Real(0.01, 1.0, prior="log-uniform"),
        "reduce_dims__n_components": Categorical([2, 0.95]),
        "rfe__n_features_to_select": Integer(5, 15)
    },
    # SVR
    {
        "RBS__M": Integer(min_val, max_val),
        "reg": Categorical([SVR()]),
        "reg__C": Real(0.1, 10.0, prior="log-uniform"),
        "reg__kernel": Categorical(['rbf', 'linear']),
        "reduce_dims__n_components": Categorical([2, 0.95]),
        "rfe__n_features_to_select": Integer(5, 15)
    }
]
#%%

#cv = RepeatedKFold(n_splits=20, n_repeats=5, random_state=0)
#cv=3
#%%

pipe.fit(X[:10], Y[:10])
print(pipe.named_steps)
print(pipe.get_params().keys())
print(pipe.score(X[:10], Y[:10]))

####TESTS TO SE IF OVERFITTING IS HAPPENING
#%%

cv=2
factor= (cv-1)/cv
min_val = min(int(np.sqrt(X.shape[0]*factor)), int(X.shape[0]*factor))
max_val = max(int(np.sqrt(X.shape[0]*factor)), int(X.shape[0]*factor))
overfit_test_param_grid = [
    {
        "RBS__M": range(min_val, max_val),  
        "reg": [LinearRegression()],
        "reg__fit_intercept": [True, False],
    },
]

gridtestof = GridSearchCV(pipefs, overfit_test_param_grid, cv=cv, scoring=("r2","neg_mean_absolute_percentage_error"), n_jobs=-1, verbose=2, return_train_score=True, error_score='raise', refit="r2")
gridtestof.fit(X, Y)

print("best score CV feat sel:", gridtestof.best_score_)
print("best params:", gridtestof.best_params_)
#%%
of_test_pipe_pca_rfe = Pipeline([
    ('scale', RobustScaler()),
    ("RBS", RBFFeatures(M=10)),
    ("rfe", RFE(estimator=LinearRegression())),  # choose desired number
    ('reduce_dims', PCA(n_components=0.99)),
    ("reg", LinearRegression())
])

of_test_param_grid_pca_rfe = [
    # LinearRegression
    {
        "RBS__M": range(min_val, max_val),
        "reg": [LinearRegression()],
        "reg__fit_intercept": [True, False],
        "reduce_dims__n_components": [3, 0.95,0.99],
        'rfe__n_features_to_select': [3, 5, 7, 10],
    },
]


gridtestofpcarfe = GridSearchCV(of_test_pipe_pca_rfe, of_test_param_grid_pca_rfe, cv=cv, scoring=("r2","neg_mean_absolute_percentage_error"), n_jobs=-1, verbose=2, return_train_score=True, error_score='raise', refit="r2")
gridtestofpcarfe.fit(X, Y)

print("best score CV PCA + RFE:", gridtestofpcarfe.best_score_)
print("best params:", gridtestofpcarfe.best_params_)
##################################
#%%
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring=("r2","neg_mean_absolute_percentage_error"), n_jobs=-1, verbose=2, return_train_score=True, refit="r2")
grid.fit(X, Y)


print("best score CV no feat sel:", grid.best_score_)
print("best params:", grid.best_params_)
#%% Feature selection
gridfs = GridSearchCV(pipefs, param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=2, refit="r2", return_train_score=True)
gridfs.fit(X, Y)

print("best score CV feat sel:", gridfs.best_score_)
print("best params:", gridfs.best_params_)


#%%
gridpca = GridSearchCV(pipepca, param_grid_pca, cv=cv, scoring="r2", n_jobs=-1, verbose=2, refit="r2", return_train_score=True)
gridpca.fit(X, Y)

print("best score PCA:", gridpca.best_score_)
print("best params:", gridpca.best_params_)


#%%

gridpcarfe = GridSearchCV(pipe_pca_rfe, param_grid_pca_rfe, cv=cv, scoring="r2", n_jobs=-1, verbose=2, refit="r2", return_train_score=True)
#%%
gridpcarfe.fit(X, Y)

print("best score PCA RFE:", gridpcarfe.best_score_)
print("best params:", gridpcarfe.best_params_)


#%%

time = datetime.datetime.now()
df = pd.DataFrame(grid.cv_results_)
df.to_csv(f"/mnt/c/Users/filom/Desktop/cv_results - {time}.csv", index=False, sep=';')






#%%

print("best score CV no feat sel:", grid.best_score_)
print("best params:", grid.best_params_)

# %%
print("best score CV feat sel:", gridfs.best_score_)
print("best params:", gridfs.best_params_)

#%%
print("best score CV feat sel:", gridpca.best_score_)
print("best params:", gridpca.best_params_)
#%%
print("best score PCA RFE:", gridpcarfe.best_score_)
print("best params:", gridpcarfe.best_params_)
#%%
########################### Bayes search
n_iter = 5

#%%
bayes = BayesSearchCV(pipe, param_space, cv=cv, scoring="r2", n_jobs=-1, verbose=2, return_train_score=True,
    refit="r2", n_iter=n_iter)
bayes.fit(X, Y)

#%%
print("best score BSearch no feat sel:", bayes.best_score_)
print("best params:", bayes.best_params_)

#%%
bayesfs = BayesSearchCV(pipefs, param_space, cv=cv, scoring=("r2","neg_mean_squared_error", "neg_mean_absolute_percentage_error"), n_jobs=-1, verbose=2, return_train_score=True,
    refit="r2", n_iter=n_iter)
bayesfs.fit(X, Y)

#%%
print("best score BSearch fet sel:", bayesfs.best_score_)
print("best params:", bayesfs.best_params_)

#%%
bayespca = BayesSearchCV(pipepca, param_space_pca, cv=cv, scoring=("r2","neg_mean_squared_error", "neg_mean_absolute_percentage_error"), n_jobs=-1, verbose=2, return_train_score=True,
    refit="r2", n_iter=n_iter)
bayespca.fit(X, Y)

#%%
print("best score BSearch pca:", bayespca.best_score_)
print("best params:", bayespca.best_params_)

#%%
bayespcarfe = BayesSearchCV(pipe_pca_rfe, param_space_pca_rfe, cv=cv, scoring=("r2","neg_mean_squared_error", "neg_mean_absolute_percentage_error"), n_jobs=-1, verbose=2, return_train_score=True,
    refit="r2", n_iter=n_iter)
bayespcarfe.fit(X, Y)

#%%
print("best score BSearch pca + rfe:", bayespcarfe.best_score_)
print("best params:", bayespcarfe.best_params_)


#%%%  Tests (remove)
try:
    print("X.shape:", X.shape)
except Exception as e:
    print("X.shape error:", e)
try:
    print("Y.shape:", getattr(Y, "shape", None))
except Exception as e:
    print("Y.shape error:", e)


#%% Lasso CV

"""
# Lasso: tune alpha (set higher max_iter to avoid warnings)
{
    "RBS__M": range(min_val, max_val),
    "reg": [Lasso(max_iter=20000, tol=1e-4)],
    'reg__alpha': [1e-2, 1e-1, 1.0],
    'reg__l1_ratio': [0.1, 0.5, 0.9]
},"""

pipe_lasso = Pipeline([
    ('scaler1', RobustScaler()),
    ('RBF', RBFFeatures(M=10)),
    ('scaler2', StandardScaler()),
    ('lasso', Lasso(max_iter=20000, tol=1e-4))
])

pipe_lasso_pca = Pipeline([
    ('scaler1', RobustScaler()),
    ('RBF', RBFFeatures(M=10)),
    ('pca', PCA()),              
    ('scaler2', StandardScaler()),
    ('lasso', Lasso(max_iter=20000, tol=1e-4))
])

pipe_lasso_fs = Pipeline([
    ("scaler", RobustScaler()),
    ("RBF", RBFFeatures(M=10)), 
    ('drop_corr', DropHighlyCorrelated(threshold=0.99)),
    ('drop_low_target', DropLowTargetCorrelation(threshold=0.01)),     # placeholder M (will be overridden)
    ('lasso', Lasso(max_iter=20000, tol=1e-4)  )
])    

param_grid_lasso = {
    'RBF__M': range(min_val, 1000),
    'lasso__alpha': [1e-4,1e-3, 1e-1, 1]
}

param_grid_lasso_pca = {
    'rbf__M': range(min_val, 1000),
    'pca__n_components': [5, 10, 15],
    'lasso__alpha': [1e-4,1e-3, 1e-1, 1]
}

#%%

gridlasso = GridSearchCV(pipe_lasso, param_grid_lasso, cv=cv, scoring="r2", n_jobs=-1, verbose=2, refit="r2", return_train_score=True)
gridlasso.fit(X, Y)

#%%
print("best score Lasso:", gridlasso.best_score_)
print("best params:", gridlasso.best_params_)

#%%
gridlassofs = GridSearchCV(pipe_lasso_fs, param_grid_lasso, cv=cv, scoring="r2", n_jobs=-1, verbose=2, refit="r2", return_train_score=True)
gridlassofs.fit(X, Y)
#%%
print("best score Lasso:", gridlassofs.best_score_)
print("best params:", gridlassofs.best_params_)

#%%
#%% Plotting train scores vs test scores 

def plot_train_vs_test(df):

    train_r2_cols = [c for c in df.columns if c.startswith("split") and c.endswith("_train_score")]
    test_r2_cols  = [c for c in df.columns if c.startswith("split") and c.endswith("_test_score")]

    # Compute mean across folds for each parameter setting
    df["mean_train_r2"] = df[train_r2_cols].mean(axis=1)
    df["mean_test_r2"]  = df[test_r2_cols].mean(axis=1)

    df["delta_r2"] = df["mean_train_r2"] - df["mean_test_r2"]

    #%%

    plt.figure(figsize=(6,6))
    r2plot = plt.scatter(df["mean_train_r2"], df["mean_test_r2"], c=df["delta_r2"], cmap="coolwarm", s=2)
    plt.plot([0,1],[0,1], "k--", label="Train = Test")  # reference line
    plt.legend()

    cbar = plt.colorbar(r2plot)
    cbar.set_label("ΔR² = Train - Test")

    plt.xlim(0.6, 1)
    plt.ylim(0.5, 1)

    plt.xlabel("Mean Train R²")
    plt.ylabel("Mean Test R²")
    plt.title("Train vs Test R² per GridSearchCV candidate")
    plt.legend()
    plt.show()

    desktop_folder = "/mnt/c/Users/filom/Desktop/Project1 ML"
    file_path = os.path.join(desktop_folder, f"train_vs_test_r2 {time}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure saved to:", file_path)


plot_train_vs_test(df)
# %%

def plot_overfitting_limit(grid, figsize=(12, 6), save_path=None):
    """
    Plot train and CV scores vs number of RBF features for each model in a GridSearchCV.
    
    Parameters:
    - grid : fitted GridSearchCV object
    - figsize : tuple, size of the figure
    - save_path : str or None, file path to save the figure
    """
    results = grid.cv_results_
    
    # Convert estimator objects to class names
    model_names = np.array([type(param['reg']).__name__ for param in results['params']])
    models = np.unique(model_names)
    
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i, model in enumerate(models):
        ax = axes[i]
        mask = model_names == model
        
        # Extract RBF numbers and scores
        M_values = np.array([param.get('RBS__M', 0) for param in results['params']])[mask]
        train_scores = results['mean_train_score'][mask]
        test_scores = results['mean_test_score'][mask]
        
        # Sort by RBF number
        sorted_idx = np.argsort(M_values)
        M_values = M_values[sorted_idx]
        train_scores = train_scores[sorted_idx]
        test_scores = test_scores[sorted_idx]
        
        # Plot
        ax.plot(M_values, train_scores, color="blue", linestyle='--', label='Train')
        ax.plot(M_values, test_scores, color="red", label='CV')
        ax.set_title(model)
        ax.set_xlabel('Number of RBF features (M)')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend()
    
    # Remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()

    desktop_folder = "/mnt/c/Users/filom/Desktop/Project1 ML"
    file_path = os.path.join(desktop_folder, f"train_test_vs_r2 {time}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure saved to:", file_path)

plot_overfitting_limit(grid)
# %%
plot_overfitting_limit(gridfs)
# %%
plot_overfitting_limit(gridpca)

#%%
print(min_val, max_val)
# %%


def plot_score_train_test_vs_rbfs(grid, model_type, figsize=(14, 8), save_path=None, desktop_save=True):
    """
    Plot train and CV scores vs number of RBF features for one selected model type.
    Creates one subplot per unique hyperparameter configuration for that model.

    Parameters
    ----------
    grid : fitted GridSearchCV object
        The result of GridSearchCV.fit(...)
    model_type : str or class
        Model to plot. Accepts 'LinearRegression', 'Ridge', 'PLSRegression', 'SVR'
        or the actual estimator class (e.g. LinearRegression).
    figsize : tuple
        Figure size.
    save_path : str or None
        If provided, save final figure to this path.
    desktop_save : bool
        If True also saves a copy to desktop path used in previous conversation.
    """
    # Normalize model_type to string class name
    if isinstance(model_type, str):
        model_name = model_type
    else:
        model_name = model_type.__name__

    results = grid.cv_results_
    params_list = results['params']

    # Build array of model class names for each param set
    param_model_names = np.array([type(p['reg']).__name__ for p in params_list])

    # Filter only rows corresponding to requested model
    mask_model = param_model_names == model_name
    if not np.any(mask_model):
        raise ValueError(f"No results found for model '{model_name}' in grid.cv_results_. "
                         f"Available models: {np.unique(param_model_names)}")

    # Extract only the relevant entries
    relevant_params = [params_list[i] for i in np.where(mask_model)[0]]
    relevant_train = np.array(results['mean_train_score'])[mask_model]
    relevant_test = np.array(results['mean_test_score'])[mask_model]

    # For each param dict, extract RBS__M and reg__* hyperparameters
    Ms = np.array([p.get('RBS__M', 0) for p in relevant_params])

    # Create a key that groups by hyperparameters under 'reg__*'
    def reg_hyperparam_key(p):
        # collect reg__... keys and their values, sorted by key name
        items = [(k[len('reg__'):], p[k]) for k in p.keys() if k.startswith('reg__')]
        # If there are no reg__ keys, set to empty tuple (works for LinearRegression with only default)
        if not items:
            return tuple()
        items_sorted = tuple(sorted(items))
        return items_sorted

    keys = [reg_hyperparam_key(p) for p in relevant_params]
    unique_keys = []
    key_to_indices = {}
    for idx, k in enumerate(keys):
        if k not in key_to_indices:
            key_to_indices[k] = []
            unique_keys.append(k)
        key_to_indices[k].append(idx)

    n_plots = len(unique_keys)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, k in enumerate(unique_keys):
        ax = axes[i]
        idxs = key_to_indices[k]
        M_vals = Ms[idxs]
        train_vals = relevant_train[idxs]
        test_vals = relevant_test[idxs]

        # Sort by M
        sorted_idx = np.argsort(M_vals)
        M_sorted = M_vals[sorted_idx]
        train_sorted = train_vals[sorted_idx]
        test_sorted = test_vals[sorted_idx]

        ax.plot(M_sorted, train_sorted, linestyle='--', label='Train')
        ax.plot(M_sorted, test_sorted, label='CV')
        ax.set_xlabel('Number of RBF features (M)')
        ax.set_ylabel('Score')
        ax.grid(True)

        # Human-readable label from hyperparam key
        if k == tuple():
            label = "default / no reg__ params"
        else:
            label = ", ".join(f"{name}={val}" for name, val in k)
        ax.set_title(label)
        ax.legend()

    # remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"{model_name} — Train vs CV scores by RBF count and hyperparam setting", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save if requested
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("Saved figure to:", save_path)

    # Optional desktop save (keeps previous behaviour)
    if desktop_save:
        try:
            desktop_folder = "/mnt/c/Users/filom/Desktop/Project1 ML"
            os.makedirs(desktop_folder, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(desktop_folder, f"{model_name}_train_test_vs_r2_{timestamp}.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            print("Figure saved to:", file_path)
        except Exception as e:
            print("Desktop save failed:", e)

    plt.show()
    plt.close(fig)
#%%
# Example usage:

plot_score_train_test_vs_rbfs(gridfs, 'LinearRegression')  
plot_score_train_test_vs_rbfs(gridfs, 'Ridge')  
plot_score_train_test_vs_rbfs(gridfs, 'SVR')  
plot_score_train_test_vs_rbfs(gridfs, 'PLSRegression')  

# %%
