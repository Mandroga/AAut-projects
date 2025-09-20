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

#%%
# Toy dataset
#X = np.linspace(0, 10, 20).reshape(-1, 1)  # 20 points from 0 to 10
#Y = np.sin(X).ravel()                     # target is sin(x) (nonlinear!)


#Real Dataset
X = np.load("X_train.npy")
Y= np.load('Y_train.npy')

# Plot
for i in range(6):
    plt.figure()
    plt.scatter(X[:,i], Y, c='k', marker='o')
    #plt.title("Toy dataset for RBF + Linear Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Feature {i} vs Target")
plt.show()

#%%
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

np.min(np.abs(y_train)), np.min(np.abs(y_val))
# %%

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    Works with negative and positive values
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_pred - y_true)/np.sum((np.abs(y_true) + np.abs(y_pred)))    
    #return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

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

metric_names = ['MSE', 'bias-sensitive SMAPE', 'R2']
predictions = []
scores_rbf = []
min_val= min(int(np.sqrt(X_train.shape[0])), int(X_train.shape[0]/3))
max_val = max(int(np.sqrt(X_train.shape[0])), int(X_train.shape[0]/3))
for i in range(min_val,max_val):  # Test different numbers of RBFs):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("RBS", RBFFeatures(M=i+1)),
        ("linreg", LinearRegression())
        ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    predictions.append(y_pred)
    MSE= mean_squared_error(y_val, y_pred)
    SMAPE= smape(y_val, y_pred)
    R2= r2_score(y_val, y_pred) 
    scores_rbf.append((i, (MSE, SMAPE, R2)))
    #print(f'RBF M={M}, MSE: {mean_squared_error(y_val, y_pred)}, MAPE: {mean_absolute_percentage_error(y_val, y_pred)}, R2: {r2_score(y_val, y_pred)}')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, name in enumerate(metric_names):
    axes[i].scatter([pt[0]+1 for pt in scores_rbf], [pt[1][i] for pt in scores_rbf], c='#17becf')
    axes[i].set_xlabel('Number of RBFs (M)')
    axes[i].set_ylabel(name)
    axes[i].set_title(f'RBF Model Performance: {name}')

plt.tight_layout()
plt.show()

# %%