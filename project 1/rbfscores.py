#%%

%matplotlib inline

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import ParameterGrid  
from sklearn.model_selection import GridSearchCV 

#%%  

# Dataset
X = np.load("X_train.npy")
Y= np.load('y_train.npy')

# Plot
for i in range(6):
    plt.figure()
    plt.scatter(X[:,i], Y, c='k', marker='o')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Feature {i} vs Target")
plt.show()

#%%
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

np.min(np.abs(y_train)), np.min(np.abs(y_val))
# %%

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
            # Compute the spread (sigma) as the average distance between centroids
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

metric_names = ['MSE', "SMAPE", 'bias-sensitive SMAPE', 'R2']
 #%% Hyperparameter Tuning and Model Evaluation
scores_rbf = []  
metrics = {
    "MSE": mean_squared_error,
    "SMAPE": smape,
    "bias-sensitive SMAPE": biasssmape,
    "R2": r2_score
}


scores_rbf = []
metrics = {"MSE": mean_squared_error, "SMAPE": smape, "bias-sensitive SMAPE": biasssmape, "R2": r2_score}

# calcula range para M
min_val = min(int(np.sqrt(X_train.shape[0])), int(X_train.shape[0]/3))
max_val = max(int(np.sqrt(X_train.shape[0])), int(X_train.shape[0]/3))

# grids separados: LinearRegression (sem alpha), Ridge/Lasso (com alphas diferentes)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("RBS", RBFFeatures(M=10)),      # placeholder M (will be overridden)
    ("reg", LinearRegression())      # placeholder model (overridden)
])

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
    # Lasso: tune alpha (set higher max_iter to avoid warnings)
    {
        "RBS__M": range(min_val, max_val),
        "reg": [Lasso(max_iter=10000)],
        "reg__alpha": [0.001, 0.005, 0.01],
    }
]

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=2)
grid.fit(X, Y)

print("best score:", grid.best_score_)
print("best params:", grid.best_params_)

#%% Plotting Results

import seaborn as sns
import matplotlib.pyplot as plt


import pandas as pd

# Extract results from GridSearchCV
results = pd.DataFrame(grid.cv_results_)

# Filter only the RBF M parameter and mean test score
plt.figure()
for reg_name in ['LinearRegression', 'Ridge', 'Lasso']:
    mask = results['param_reg'].apply(lambda x: x.__class__.__name__) == reg_name
    plt.plot(results.loc[mask, 'param_RBS__M'],
             results.loc[mask, 'mean_test_score'],
             label=reg_name)

plt.xlabel("Number of RBF kernels (M)")
plt.ylabel("Mean CV R2 score")
plt.title("Performance vs Number of RBF kernels")
plt.legend()
plt.show()



# %%