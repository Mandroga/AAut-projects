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

#%%  

# Dataset
X = np.load("X_train.npy")
Y= np.load('Y_train.npy')

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
param_grid = [
    {"M": range(min_val, max_val), "model": [LinearRegression()], "alpha": [None]},
    {"M": range(min_val, max_val), "model": [Ridge()], "alpha": [0.01, 0.1, 1.0]},
    {"M": range(min_val, max_val), "model": [Lasso()], "alpha": [0.001, 0.005, 0.01]}
]

for params in ParameterGrid(param_grid):
    M_val = params["M"]
    model_instance = params["model"]
    alpha = params["alpha"]

    # só aplica alpha se fizer sentido
    if alpha is not None:
        model_instance.set_params(alpha=alpha)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("RBS", RBFFeatures(M=M_val)),
        ("reg", model_instance)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    # guarda métricas
    for name, func in metrics.items():
        score = func(y_val, y_pred)
        scores_rbf.append((
            M_val,
            type(model_instance).__name__,
            alpha if alpha is not None else "N/A",  # LinearRegression não tem alpha
            name,
            score
        ))

df_scores = pd.DataFrame(scores_rbf, columns=["n_rbf", "model", "alpha", "metric", "score"])

#%% Plotting Results

import seaborn as sns
import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, len(metric_names), figsize=(18, 5), sharey=False)

for ax, metric in zip(axes, metric_names):
    subset = df_scores[df_scores["metric"] == metric].copy()
    
    # cria coluna combinando modelo + alpha
    subset["model_alpha"] = subset["model"] + " (alpha=" + subset["alpha"].astype(str) + ")"
    
    sns.lineplot(
        data=subset,
        x="n_rbf",
        y="score",
        hue="model_alpha",
        ax=ax
    )
    
    ax.set_xlabel("Number of RBFs (M)")
    ax.set_ylabel(metric)
    ax.set_title(f"RBF Model Performance: {metric}")

# remove legend de todos os eixos
for ax in axes:
    ax.get_legend().remove()

# cria legenda única abaixo do gráfico
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.25))

plt.tight_layout()
plt.show()


# %%