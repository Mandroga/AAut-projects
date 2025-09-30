#%%
%run imports_data.py

time = datetime.datetime.now()

#%%
%matplotlib inline

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

#%%

#Load Data
X = np.load("X_train.npy")
Y= np.load('y_train.npy')


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.5, test_size=0.5, random_state=42)
####TESTS TO SE IF OVERFITTING IS HAPPENING
# %%   class RBFFeatures                         
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

#%%
from sklearn.model_selection import ShuffleSplit    

pipe_separate_test = Pipeline([
    ("scaler", RobustScaler()),
    ("RBS", RBFFeatures(M=193)),     
    ("reg", LinearRegression(fit_intercept=False))
])


pipe_separate_test.fit(X_train, Y_train)
score = pipe_separate_test.score(X_test, Y_test)
#%%
print("score no feat sel:",score)
print("training score:", pipe_separate_test.score(X_train, Y_train))
#%%
pipe_separate_test_fs = Pipeline([
    ("scaler", RobustScaler()),
    ("RBS", RBFFeatures(M=193)), 
    ('drop_corr', DropHighlyCorrelated(threshold=0.99)),
    ('drop_low_target', DropLowTargetCorrelation(threshold=0.01)),      
    ("reg", LinearRegression(fit_intercept=False))
])

pipe_separate_test.fit(X_train, Y_train)
score_fs = pipe_separate_test.score(X_test, Y_test)
#%%
print("score feat sel:",score_fs)
print("training score:", pipe_separate_test.score(X_train, Y_train))
#%%

#%%
pipe_separate_test_of_pca_rfe = Pipeline([
    ('scale', RobustScaler()),
    ("RBS", RBFFeatures(M=193)),
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