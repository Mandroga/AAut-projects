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


min_val = min(int(np.sqrt(X.shape[0]*0.7)), int(X.shape[0]*0.7))
max_val = max(int(np.sqrt(X.shape[0])), int(X.shape[0]))
rfe_cv_folds = 5

pipe = Pipeline([
    ("scaler", RobustScaler()),
  
    ("RBS", RBFFeatures(M=10)),      # placeholder M (overridden)
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

pipefs = Pipeline([
    ("scaler", RobustScaler()),
    ("RBS", RBFFeatures(M=10)), 
    ('drop_corr', DropHighlyCorrelated(threshold=0.99)),
    ('drop_low_target', DropLowTargetCorrelation(threshold=0.01)),     # placeholder M (will be overridden)
    ("reg", LinearRegression())      # placeholder model (overridden)
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
max_val = min(max_val, 300)  # limit max M to 30 for computational reasons
# be careful: range(a, b) excludes b — use b+1 if you want inclusive max
param_grid = [
    # LinearRegression: no alpha
    {
        "RBS__M": range(min_val, max_val),   # or range(min_val, max_val+1)
        "reg": [LinearRegression()],
        # example LR-specific params if you want:
        "reg__fit_intercept": [True, False],
    },
    # Ridge: tune alpha
    {
        "RBS__M": range(min_val, max_val),
        "reg": [Ridge(max_iter=5000)],
        "reg__alpha": [0.01, 0.1, 1.0],
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
        "reg__C": [0.1, 1.0, 10.0],
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
        "reduce_dims__n_components": [5, 15, 0.95]  # can be int or float for explained variance
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
    # PLS
    {
        "RBS__M": range(min_val, max_val),
        "reg": [PLSRegression(max_iter=5000)],
        "reg__n_components": range(2, min(10, X.shape[1])),
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
        "reduce_dims__n_components": Categorical([5, 15, 0.95]),
        "rfe__n_features_to_select": Integer(5, 15)
    },
    # Ridge
    {
        "RBS__M": Integer(min_val, max_val),
        "reg": Categorical([Ridge(max_iter=5000)]),
        "reg__alpha": Real(0.01, 1.0, prior="log-uniform"),
        "reduce_dims__n_components": Categorical([5, 15, 0.95]),
        "rfe__n_features_to_select": Integer(5, 15)
    },
    # PLS
    {
        "RBS__M": Integer(min_val, max_val),
        "reg": Categorical([PLSRegression(max_iter=5000)]),
        "reg__n_components": Integer(2, min(10, X.shape[1])-1),
        "reduce_dims__n_components": Categorical([5, 15, 0.95]),
        "rfe__n_features_to_select": Integer(5, 15)
    },
    # SVR
    {
        "RBS__M": Integer(min_val, max_val),
        "reg": Categorical([SVR()]),
        "reg__C": Real(0.1, 10.0, prior="log-uniform"),
        "reg__kernel": Categorical(['rbf', 'linear']),
        "reduce_dims__n_components": Categorical([5, 15, 0.95]),
        "rfe__n_features_to_select": Integer(5, 15)
    }
]
#%%

#cv = RepeatedKFold(n_splits=20, n_repeats=5, random_state=0)
cv=3
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
gridpca.fit(X, Y)

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
########################### Bayes search
n_iter = 5
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
print("best score BSearch pca:", bayespcarfe.best_score_)
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
    ('rbf', RBFFeatures(M=10)),
    ('scaler2', StandardScaler()),
    ('lasso', Lasso(max_iter=20000, tol=1e-4))
])

pipe_lasso_pca = Pipeline([
    ('scaler1', RobustScaler()),
    ('rbf', RBFFeatures(M=10)),
    ('pca', PCA()),              
    ('scaler2', StandardScaler()),
    ('lasso', Lasso(max_iter=20000, tol=1e-4))
])

pipe_lasso_fs = Pipeline([
    ("scaler", RobustScaler()),
    ("RBS", RBFFeatures(M=10)), 
    ('drop_corr', DropHighlyCorrelated(threshold=0.99)),
    ('drop_low_target', DropLowTargetCorrelation(threshold=0.01)),     # placeholder M (will be overridden)
    ('lasso', Lasso(max_iter=20000, tol=1e-4)  )
])    

param_grid_lasso = {
    'rbf__M': [5, 10, 15],
    'lasso__alpha': [1e-3, 1e-2, 1e-1, 1]
}

param_grid_lasso_pca = {
    'rbf__M': [5, 10, 15],
    'pca__n_components': [5, 10, 15],
    'lasso__alpha': [ 1e-2, 1e-1, 1]
}



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

def plot_overfitting_limit(grid):

    df = pd.DataFrame(grid.cv_results_)

    # Unique models
    models = np.unique(results['param_reg'])

    plt.figure(figsize=(8, 5))

    for model in models:
        # Mask for this model
        mask = results['param_reg'] == model
        
        # Extract number of RBF features and scores
        M_values = np.array([param['RBS__M'] for param in results['params']])[mask]
        test_scores = results['mean_test_score'][mask]
        train_scores = results['mean_train_score'][mask]
        
        # Sort by RBF features
        sorted_idx = np.argsort(M_values)
        M_values = M_values[sorted_idx]
        test_scores = test_scores[sorted_idx]
        train_scores = train_scores[sorted_idx]
        
        # Plot test and train scores
        plt.plot(M_values, test_scores, marker='o', label=f'{model} (CV)')
        plt.plot(M_values, train_scores, marker='x', linestyle='--', label=f'{model} (Train)')

    plt.xlabel('Number of RBF features (M)')
    plt.ylabel('Score')
    plt.title('Train vs CV Score vs Number of RBF Features')
    plt.legend()
    plt.grid(True)
    plt.show()

    desktop_folder = "/mnt/c/Users/filom/Desktop/Project1 ML"
    file_path = os.path.join(desktop_folder, f"train_vs_test_r2 - {RegressorName}, {time}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Figure saved to:", file_path)

plot_overfitting_limit(grid)
# %%
