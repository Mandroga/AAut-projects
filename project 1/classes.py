import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
import pandas as pd


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