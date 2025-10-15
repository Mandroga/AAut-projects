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
# %%
pre_processor = Pipeline([
    ("DropDuplicate", DropDuplicateFeatures()),
    #("DropConstant", DropConstantFeatures(tol=0.95)),
    #("DropCorrelated", SmartCorrelatedSelection(method="pearson", threshold=0.8, selection_method="variance" ))
    ("DropCorrelated", DropCorrelatedFeatures(threshold=0.9)),
    #("stdscaler", StandardScaler())
    ])
X_pre = pre_processor.fit_transform(X_pre)
print('pre2')


# %%
print(X_pre.shape)
# %% classifier

pls_model = Pipeline([
    ("pls", PLSRegression(n_components=5, scale=True)),  # transformer step
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