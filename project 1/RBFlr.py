#%%
%run imports_data.py

#%%
%matplotlib inline

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

#%%  

time = datetime.datetime.now()
# Dataset loading

X = np.load("X_train.npy")
Y= np.load('y_train.npy')

#%% Definition of RBF class (a built in was not found in sklearn)                   
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
"""COMMENT HERE"""
cv=2
factor= (cv-1)/cv
min_n_cent = min(int(np.sqrt(X.shape[0]*factor)), int(X.shape[0]*factor))
max_n_cent = max(int(np.sqrt(X.shape[0]*factor)), int(X.shape[0]*factor))

#Pipeline for GridSearchCV with RBF + Linear Regression

pipe = Pipeline([
    ("scaler", RobustScaler()), #scaler robust to outliers
    ("RBS", RBFFeatures(M=10)),    
    ("reg", LinearRegression())      
])

#Pipeline for GridSearchCV with RBF + Linear Regression + Feature Selection

pipe_fs = Pipeline([
    ("scaler", RobustScaler()),
    ("RBS", RBFFeatures(M=10)), 
    ('drop_corr', DropHighlyCorrelated(threshold=0.99)),
    ('drop_low_target', DropLowTargetCorrelation(threshold=0.01)),     
    ("reg", LinearRegression())     
])

param_grid = [
    {
        "RBS__M": range(min_n_cent, max_n_cent),   # or range(min_n_cent, max_n_cent+1)
        "reg": [LinearRegression()],
        "reg__fit_intercept": [True, False],
    },
    # Ridge
    {
        "RBS__M": range(min_n_cent, max_n_cent),
        "reg": [Ridge(max_iter=5000)],
        "reg__alpha": [0.001,0.005,0.01],
    },
    #PLS
    {
        "RBS__M": range(min_n_cent, max_n_cent),
        "reg": [PLSRegression(max_iter=5000)],
        "reg__n_components": range(2, min(10, X.shape[1]))
    },
    # SVR
    {
        "RBS__M": range(min_n_cent, max_n_cent),
        "reg": [SVR()],
        "reg__C": [10.0,12.0,14.0],
        "reg__kernel": ['rbf', 'linear']
    }
]

#%%
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring=("r2","neg_mean_absolute_percentage_error"), n_jobs=-1, verbose=2, return_train_score=True, refit="r2")
grid.fit(X, Y)


print("best score CV no feature selection:", grid.best_score_)
print("best params:", grid.best_params_)
#%% Feature selection
grid_fs = GridSearchCV(pipe_fs, param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=2, return_train_score=True)
grid_fs.fit(X, Y)

print("best score CV with feature selection:", grid_fs.best_score_)
print("best params:", grid_fs.best_params_)

################################ PCA
#%% Testing RBF + PCA + RFE + OLS with GridSearchCV 

pipe_pca_rfe = Pipeline([
    ('scale', RobustScaler()),
    ("RBS", RBFFeatures(M=10)),
    ("rfe", RFE(estimator=LinearRegression())),  # choose desired number
    ('reduce_dims', PCA(n_components=0.99)),
    ("reg", LinearRegression())
])

param_grid_pca_rfe = [
    {
        "RBS__M": range(min_n_cent, max_n_cent),
        "reg": [LinearRegression()],
        "reg__fit_intercept": [True, False],
        "reduce_dims__n_components": [3, 0.95, 0.99],
        'rfe__n_features_to_select': [3, 5, 7],
    },
]


grid_pca_rfe = GridSearchCV(pipe_pca_rfe, param_grid_pca_rfe, cv=cv, scoring=("r2","neg_mean_absolute_percentage_error"), n_jobs=-1, verbose=2, return_train_score=True, error_score='raise', refit="r2")
grid_pca_rfe.fit(X, Y)

print("best score CV PCA + RFE:", grid_pca_rfe.best_score_)
print("best params:", grid_pca_rfe.best_params_)
##################################

#%% Lasso CV

pipe_lasso = Pipeline([
    ('scaler1', RobustScaler()),
    ('RBF', RBFFeatures(M=10)),
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
    'RBF__M': range(min_n_cent, 1000),
    'lasso__alpha': [1e-4,1e-3, 1e-1, 1]
}

#%%

grid_lasso = GridSearchCV(pipe_lasso, param_grid_lasso, cv=cv, scoring="r2", n_jobs=-1, verbose=2, return_train_score=True)
grid_lasso.fit(X, Y)

#%%
print("best score Lasso:", grid_lasso.best_score_)
print("best params:", grid_lasso.best_params_)

#%%
grid_lasso_fs = GridSearchCV(pipe_lasso_fs, param_grid_lasso, cv=cv, scoring="r2", n_jobs=-1, verbose=2, refit="r2", return_train_score=True)
grid_lasso_fs.fit(X, Y)
#%%
print("best score Lasso:", grid_lasso_fs.best_score_)
print("best params:", grid_lasso_fs.best_params_)

#%% Plotting train scores vs test scores 


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
        If True also saves a copy to desktop path.
    """
    # Normalizing model_type to string class name
    if isinstance(model_type, str):
        model_name = model_type
    else:
        model_name = model_type.__name__

    results = grid.cv_results_
    params_list = results['params']

    # Building array of model class names for each param set
    param_model_names = np.array([type(p['reg']).__name__ for p in params_list])

    # Filtering only rows corresponding to requested model
    mask_model = param_model_names == model_name
    if not np.any(mask_model):
        raise ValueError(f"No results found for model '{model_name}' in grid.cv_results_. "
                         f"Available models: {np.unique(param_model_names)}")

    # Extracting only the relevant entries
    relevant_params = [params_list[i] for i in np.where(mask_model)[0]]
    relevant_train = np.array(results['mean_train_score'])[mask_model]
    relevant_test = np.array(results['mean_test_score'])[mask_model]

    # Extracting, for each param dict, RBS__M and reg__* hyperparameters
    Ms = np.array([p.get('RBS__M', 0) for p in relevant_params])

    # Creation of a key that groups by hyperparameters under 'reg__*'
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

        # Sorting by M
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

    # removing unused subplots
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
for i in ['LinearRegression', 'Ridge', 'SVR', 'PLSRegression']:
    plot_score_train_test_vs_rbfs(grid_fs, i)  

# %%
