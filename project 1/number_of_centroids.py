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
cv=5
factor= (cv-1)/cv
min_n_cent = min(int(np.sqrt(X.shape[0]*factor)), int(X.shape[0]*factor))
max_n_cent = max(int(np.sqrt(X.shape[0]*factor)), int(X.shape[0]*factor))

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
]


#%% Feature selection
grid_fs = GridSearchCV(pipe_fs, param_grid, cv=cv, scoring="r2", n_jobs=-1, verbose=2, return_train_score=True)
best_centroids = []
for i in range(140, 600, 40):
    # Random subset of size i
    X_sub, _, Y_sub, _ = train_test_split(X, Y, train_size = i/700, test_size=(700-i)/700, random_state=None)

    grid_fs.fit(X_sub, Y_sub)

    print("best score CV:", grid_fs.best_score_, "size:", i)
    best_centroids.append(grid_fs.best_estimator_)
# %%
