# %% imports
from imports3 import *
# %% load data
with open("Xtrain2.pkl", "rb") as f:
    X = pickle.load(f)
Y= np.load("Ytrain2.npy")

# %% preprocess
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    DropCorrelatedFeatures,
    SmartCorrelatedSelection
)


class preprocess_data_pls(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_transformed = []

        # Loop through rows (can vectorize later if large)
        for i, row in X.iterrows():
            patient_id = row['Patient_Id']
            exercise_id = row['Exercise_Id']
            X_ss = row['Skeleton_Sequence']
            indexes_drop = [18,16,22,15,21,17,6,5,4,1,2,3,28,30,27,29]
            cols_drop = [i for j in indexes_drop for i in [2*j,2*j+1] ]
            X_ss = np.delete(X_ss, cols_drop, axis=1)

            flat = np.ravel(X_ss)  # or .flatten()

            X_transformed.append({
                "Patient_Id": patient_id,
                "Exercise_Id": exercise_id,
                **{f"f{j}": v for j, v in enumerate(flat)}
            })

        df = pd.DataFrame(X_transformed)

        # One-hot encode categorical columns
        df = pd.get_dummies(df, columns=['Patient_Id', 'Exercise_Id'], dtype=int)
        df.fillna(0, inplace=True)  # Handle any NaNs from one-hot encoding
        return df

active_classes = np.where(Y==1)[0]+1
Y_ = X['Patient_Id'].isin(active_classes).astype(int)
X_pre = preprocess_data_pls().fit_transform(X)
groups_all = X['Patient_Id'].to_numpy()
print('pre1')
# %% pre2
from scipy import sparse
class FastNearConstantFilter(BaseEstimator, TransformerMixin):
    """
    Fast remover of constant / near-constant features.

    Strategy (in this order, vectorized):
      1) True constants: max - min == 0  (dense) OR all-zero columns (sparse)
      2) Zero-dominant proxy: mean(X==0) >= tol   (dense) OR 1 - nnz/n >= tol (sparse)
      3) Robust spread (MAD): (MAD <= mad_eps) OR (MAD / (|median|+eps) <= rcv_thresh)
      4) Optional exact dominant-value check on survivors (slow loop, but small set)

    Parameters
    ----------
    tol : float, default=0.95
        Threshold for dominant value proportion (used for zero-dominant and exact checks).
    use_zero_proxy : bool, default=True
        If True, treat 0 as the candidate dominant value (fast proxy).
    mad_eps : float, default=1e-12
        Absolute flatness threshold for MAD (dense only).
    rcv_thresh : float, default=1e-3
        Robust coefficient of variation threshold: MAD / (|median|+eps).
    exact_on_survivors : bool, default=False
        If True, run exact dominant-value proportion on remaining columns.
    max_exact_cols : int or None, default=5000
        Hard cap on number of survivor columns for the exact pass. None = no cap.
    treat_nan_as_value : bool, default=False
        If True, NaN counts as a value in the exact dominant check (dense only).
        For zero-proxy and MAD, NaNs are ignored where applicable.
    """

    def __init__(self,
                 tol=0.95,
                 use_zero_proxy=True,
                 mad_eps=1e-12,
                 rcv_thresh=1e-3,
                 exact_on_survivors=False,
                 max_exact_cols=5000,
                 treat_nan_as_value=False):
        self.tol = tol
        self.use_zero_proxy = use_zero_proxy
        self.mad_eps = mad_eps
        self.rcv_thresh = rcv_thresh
        self.exact_on_survivors = exact_on_survivors
        self.max_exact_cols = max_exact_cols
        self.treat_nan_as_value = treat_nan_as_value

    def fit(self, X, y=None):
        X = self._validate_X(X)
        n, p = X.shape
        self.n_features_in_ = p

        # 1) True constants
        if sparse.issparse(X):
            # all-zero columns are constant zero
            nnz = X.getnnz(axis=0)
            const_mask = (nnz == 0)
        else:
            # max-min == 0 (NaNs ignored in reductions)
            col_max = np.nanmax(X, axis=0)
            col_min = np.nanmin(X, axis=0)
            const_mask = (col_max - col_min) == 0

        drop_mask = const_mask.copy()

        # 2) Zero-dominant proxy
        if self.use_zero_proxy:
            if sparse.issparse(X):
                zero_ratio = 1.0 - (X.getnnz(axis=0) / float(n))
            else:
                zero_ratio = np.mean(X == 0, axis=0)
            drop_mask |= (zero_ratio >= self.tol)

        # 3) Robust spread via MAD (dense only)
        if not sparse.issparse(X):
            med = np.nanmedian(X, axis=0)
            mad = np.nanmedian(np.abs(X - med), axis=0)
            low_abs = (mad <= self.mad_eps)
            rcv = mad / (np.abs(med) + 1e-12)
            low_rel = (rcv <= self.rcv_thresh)
            drop_mask |= (low_abs | low_rel)

        # 4) Optional exact dominant-value check on survivors (dense only)
        if self.exact_on_survivors and not sparse.issparse(X):
            survivors = np.flatnonzero(~drop_mask)
            if (self.max_exact_cols is None) or (len(survivors) <= self.max_exact_cols):
                exact_drop = np.zeros_like(drop_mask, dtype=bool)
                for j in survivors:
                    col = X[:, j]
                    if self.treat_nan_as_value:
                        # Include NaN as a category: replace NaN with a sentinel
                        col = col.copy()
                        col[np.isnan(col)] = np.inf  # sentinel unlikely to appear
                    else:
                        # Drop NaNs from count/denominator
                        col = col[~np.isnan(col)]
                    if col.size == 0:
                        # all NaN -> treat as constant-like
                        exact_drop[j] = True
                        continue
                    _, counts = np.unique(col, return_counts=True)
                    if counts.max() / float(n) >= self.tol:
                        exact_drop[j] = True
                drop_mask |= exact_drop

        self.drop_mask_ = drop_mask
        self.keep_mask_ = ~drop_mask
        self.keep_indices_ = np.flatnonzero(self.keep_mask_)
        return self

    def transform(self, X):
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}."
            )
        if sparse.issparse(X):
            return X[:, self.keep_indices_]
        return X[:, self.keep_indices_]

    # ---- utilities ----
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        input_features = np.asarray(input_features)
        return input_features[self.keep_indices_]

    @staticmethod
    def _validate_X(X):
        if sparse.issparse(X):
            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()
            return X
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        return X
    
pre_processor = Pipeline([
    ("DropConstant", FastNearConstantFilter(tol=0.95,use_zero_proxy=True,exact_on_survivors=False)),
    ("DropCorrelated", SmartCorrelatedSelection(method="pearson", threshold=0.8, selection_method="variance" )),
    ("stdscaler", StandardScaler())
    ])
X_pre = pre_processor.fit_transform(X_pre)
print('pre2')
print(X_pre.shape)


# %%
print(X_pre.shape)

# %%
from sklearn.preprocessing import FunctionTransformer
pls = PLSRegression(n_components=5, scale=False)  # preproc already scaled
T = pls.fit(X_pre, Y_).transform(X_pre)
print("After PLS.transform:", T.ndim, T.shape, type(T))
# %% classifier
flatten2d = FunctionTransformer(
    lambda X: np.asarray(X).reshape(X.shape[0], -1),
    validate=False
)

pls_model = Pipeline([
    ("pls", PLSRegression(n_components=5, scale=True)), 
    ('flatten', flatten2d),
    ("clf", LogisticRegression(max_iter=100))
])

search_space = {'pls__n_components': (10, 100),  # integer valued parameter
                'clf__max_iter': (50, 200),}
# %% training bayes

scores = []
X_data = X_pre
Y_data = Y_

sgkfs = StratifiedGroupKFoldStrict(n_splits=5, shuffle=True, random_state=42)

opt = BayesSearchCV(
estimator=pls_model,
search_spaces=search_space,
n_iter=5,                                # start with ~40-80; increase if time allows
scoring=make_scorer(balanced_accuracy_score),
cv=sgkfs,
n_jobs=-1,
refit=True,
random_state=42,
verbose=2
)
opt.fit(X_data, Y_data, groups=groups_all)

best_i = opt.best_index_

# Grab all split columns
# %%
import re
cols = [c for c in opt.cv_results_.keys() if re.match(r"split\d+_test_score", c)]

# Per-fold scores for the best params
per_fold_scores = np.array([opt.cv_results_[c][best_i] for c in cols], dtype=float)
print("Per-fold test scores:", per_fold_scores)
print("Mean ± std:", per_fold_scores.mean(), "±", per_fold_scores.std())