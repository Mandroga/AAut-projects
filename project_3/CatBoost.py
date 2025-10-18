# %% imports
from sklearn.base import ClassifierMixin
from imports3 import *
# %% load data
with open("Xtrain2.pkl", "rb") as f:
    X = pickle.load(f)
Y= np.load("Ytrain2.npy")
# %% preprocess

print(X.columns)
def sliding_average(a, window):
    # compute rolling mean along axis=0 (frames)
    cumsum = np.pad(np.cumsum(a, axis=0), ((window,0),(0,0)), mode='constant')
    return (cumsum[window:] - cumsum[:-window]) / window

class preprocess_data(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.all_keypoints = {'r':[4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
            'l':[1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31]}
        self.body_parts = {'leg':{'l':[23,25,27],'r':[24,26,28]},
                           'feet':{'l':[29,31],'r':[30,32]},
                           'hand':{'l':[17,19,21],'r':[18,20,22]},
            'arm':{'l':[11,13,15],'r':[12,14,16]},
            'torso':{'l':[11,23],'r':[12,24]},
            'upper_face':{'l':[1,2,3,7],'r':[4,5,6,8]},
            'mouth':{'l':[9],'r':[10]}}
        
    def fit(self, X, y=None):
        return self
    def transform(self, X_):
        X = X_.copy()
        at = X.at
        body_parts = self.body_parts
        all_keypoints = self.all_keypoints
        agg_fs = [('mean',np.mean),('std',np.std),('median',np.median)]
        for i in range(len(X)):
            idx = X.index[i]
            #patient_id = at[idx,'Patient_Id']
           # exercise_id = at[idx,'Exercise_Id']
            X_ss = at[idx,'Skeleton_Sequence']
            #invert y
            if 1:
                X_ss[:,1::2] = -X_ss[:,1::2]
            #normalize skeleton?
            #total distances np
            if 1:
                X_ss_diff = np.diff(X_ss, axis=0)
                X_ss_diff_xy = X_ss_diff.reshape(len(X_ss_diff), 33, 2)
                X_ss_distance = np.hypot(X_ss_diff_xy[:,:,0], X_ss_diff_xy[:,:,1])
                side = {'l':{name:0 for name, f in agg_fs},'r':{name:0 for name, f in agg_fs}}
                for key in ['l', 'r']:
                    for body_part in body_parts.keys():
                        indexes = body_parts[body_part][key]
                        distances_sum = np.sum(X_ss_distance[:,indexes], axis=1)
                        for f_n, f in agg_fs:
                            total_distances_sum = f(distances_sum)
                            at[idx,f'{body_part}_{key}_distance_{f_n}'] = total_distances_sum
                            side[key][f_n] += total_distances_sum
                    for f_n, f in agg_fs:
                        total_side = side[key][f_n]
                        at[idx,f'total_{key}_distance_{f_n}'] = total_side
            #jitter
            if 1:
                smooth = sliding_average(X_ss, window=4)
                jitter = X_ss - smooth
                Xs = jitter[:,::2]
                Ys = jitter[:,1::2]
                jitter_mag = np.hypot(Xs,Ys)
                side = {'l':{name:0 for name, f in agg_fs},'r':{name:0 for name, f in agg_fs}}
                for key in ['l', 'r']:
                    for body_part in body_parts.keys():
                        indexes = body_parts[body_part][key]
                        jitter_sum = np.sum(jitter_mag[:,indexes], axis=1)
                        for f_n, f in agg_fs:
                            total_jitter_sum = f(jitter_sum)
                            at[idx,f'{body_part}_{key}_jitter_{f_n}'] = total_jitter_sum
                            side[key][f_n] += total_jitter_sum
                    for f_n, f in agg_fs:
                        total_side = side[key][f_n]
                        at[idx,f'total_{key}_jitter_{f_n}'] = total_side
                if 0:
                    for key in ['l','r']:
                        cols = all_keypoints[key]
                        jitter_sum = np.sum(jitter_mag[:,cols], axis=1)
                        for f_n, f in agg_fs:
                            total_jitter_side = f(jitter_sum)
                            X.loc[i,f'jitter_{key}_{f_n}'] = total_jitter_side
            #mena-features
            if 1:
                
        #drop SS
        if 1:
            X = X.drop('Skeleton_Sequence', axis=1)
        return X

class GroupVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf):
        self.base_clf = base_clf

    def _patient_to_row(self, X, y):
        groups = X['Patient_Id'].unique()
        mapping = dict(zip(groups, y))
        y_ = X['Patient_Id'].map(mapping).to_numpy()
        return y_

    def fit(self, X__, y__, **fit_params):
        X = X__.copy()
        y = y__.copy()
        y_ = self._patient_to_row(X, y)
        if 'eval_set' in fit_params:
            Xv__, yv__ = fit_params['eval_set']
            Xv = Xv__.copy()
            yv = yv__.copy()
            Xv_pre = preprocess_data().fit_transform(Xv)
            yv_row = self._patient_to_row(Xv_pre, yv)
            fit_params['eval_set'] = (Xv_pre, yv_row)

        self.base_clf.fit(X, y_, **fit_params)
        print('fitted')
        return self

    def predict(self, X):
        print('predict group voting')
        preds = self.base_clf.predict(X)
        df = X.copy()
        df['pred'] = preds
        print('preds')
        voted_preds = df.groupby('Patient_Id')['pred'].agg(lambda x: np.bincount(x).argmax()).values
        print(voted_preds)
        return voted_preds

active_classes = np.where(Y==1)[0]+1
Y_ = X['Patient_Id'].isin(active_classes).astype(int)
#X_pre = preprocess_data().fit_transform(X)
groups_all = X['Patient_Id'].to_numpy()

#print(X_pre)
''' 
Another idea for features - fit data to a distribution, pass distribution parameters to the model!
'''
# %% classifier
model = CatBoostClassifier(
    iterations=100,        # número de árvores
    learning_rate=0.05,    # taxa de aprendizado
    depth=8,               # profundidade das árvores
    loss_function='Logloss',
    eval_metric='BalancedAccuracy',
    random_seed=42,
    verbose=100            # mostra progresso a cada 100 iterações
)
cat_features = ['Patient_Id','Exercise_Id']

search_space = {
    "iterations": (10, 100),                 # int
    "learning_rate": (1e-2, 0.5, "log-uniform"),
    "depth": (6, 10),                          # int
   # "l2_leaf_reg": (1e-2, 20.0, "log-uniform"),
   # "min_data_in_leaf": (1, 64),               # int
    "bagging_temperature": (0.0, 1.0),         # only used when bootstrap_type="Bayesian"
    "subsample": (0.5, 1.0),                   # only used when bootstrap_type="Bernoulli"
    "colsample_bylevel": (0.5, 1.0),
    "random_strength": (0.0, 2.0),
  #  "bootstrap_type": ["Bayesian", "Bernoulli"],
    "grow_policy": ["SymmetricTree"],          # (Lossguide) is OK too; add if you want
}

# %% classifier with voting

base_model = CatBoostClassifier(
    iterations=100,        # número de árvores
    learning_rate=0.05,    # taxa de aprendizado
    depth=8,               # profundidade das árvores
    loss_function='Logloss',
    eval_metric='BalancedAccuracy',
    random_seed=42,
    verbose=10            # mostra progresso a cada 100 iterações
)
cat_features = ['Patient_Id','Exercise_Id']

search_space = {
    "iterations": (10, 100),                 # int
    "learning_rate": (1e-2, 0.5, "log-uniform"),
    "depth": (6, 10),                          # int
   # "l2_leaf_reg": (1e-2, 20.0, "log-uniform"),
   # "min_data_in_leaf": (1, 64),               # int
    "bagging_temperature": (0.0, 1.0),         # only used when bootstrap_type="Bayesian"
    "subsample": (0.5, 1.0),                   # only used when bootstrap_type="Bernoulli"
    "colsample_bylevel": (0.5, 1.0),
    "random_strength": (0.0, 2.0),
  #  "bootstrap_type": ["Bayesian", "Bernoulli"],
    "grow_policy": ["SymmetricTree"],          # (Lossguide) is OK too; add if you want
}

model = Pipeline([
    ('pre',preprocess_data()),
    ('gvc',GroupVotingClassifier(base_clf=base_model))
    ])


# %% training no tuning

X_pre = preprocess_data().fit_transform(X)
scores = []
X_data = X_pre
Y_data = Y_

sgkfs = StratifiedGroupKFoldStrict(n_splits=5, shuffle=True, random_state=42)

for iteration, (train_val_idx, test_idx) in enumerate(sgkfs.split(X_data, Y_data, groups=groups_all)):
    X_train, X_test = X_data.iloc[train_val_idx], X_data.iloc[test_idx]
    y_train, y_test = Y_data[train_val_idx], Y_data[test_idx]
    groups_train = groups_all[train_val_idx]
    groups_test = groups_all[test_idx]
    vals, counts_train = np.unique(y_train, return_counts=True)
    vals, counts_test = np.unique(y_test, return_counts=True)
    print(counts_train, counts_test)
    print(np.unique(groups_train), np.unique(groups_test))

    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), use_best_model=True)
    y_pred = model.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    scores.append(score)
    print("Balanced Accuracy:", score)
print("Per-fold test scores:", scores)
print("Mean Balanced Accuracy:", np.mean(scores), '+-', np.std(scores))

# %% training bayes

scores = []
X_data = X_pre
Y_data = Y_

sgkfs = StratifiedGroupKFoldStrict(n_splits=5, shuffle=True, random_state=42)

opt = BayesSearchCV(
estimator=model,
search_spaces=search_space,
n_iter=5,                                # start with ~40-80; increase if time allows
scoring=make_scorer(balanced_accuracy_score),
cv=sgkfs,
n_jobs=-1,
refit=True,
random_state=42,
verbose=2
)
opt.fit(X_data, Y_data, groups=groups_all, cat_features=cat_features)

best_i = opt.best_index_

# Grab all split columns
# %%
import re
cols = [c for c in opt.cv_results_.keys() if re.match(r"split\d+_test_score", c)]

# Per-fold scores for the best params
per_fold_scores = np.array([opt.cv_results_[c][best_i] for c in cols], dtype=float)
print("Per-fold test scores:", per_fold_scores)
print("Mean ± std:", per_fold_scores.mean(), "±", per_fold_scores.std())
# %% feat importance
importances = model.get_feature_importance()
feature_names = model.feature_names_

feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feat_imp.head(10))

feat_imp.head(20).plot(kind='barh', figsize=(8,6))

model.plot_tree()  # visualize trees

plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance")
plt.show()

# %% Group voting - training no tuning

scores = []
X_data = X
Y_data = Y_

sgkfs = StratifiedGroupKFoldStrict(n_splits=5, shuffle=True, random_state=42)

for iteration, (train_val_idx, test_idx) in enumerate(sgkfs.split(X_data, Y_data, groups=groups_all)):
    X_train, X_test = X_data.iloc[train_val_idx], X_data.iloc[test_idx]
    groups_train = groups_all[train_val_idx]
    groups_test = groups_all[test_idx]
    y_train, y_test = Y[np.unique(groups_train)-1], Y[np.unique(groups_test)-1]
    model.fit(X_train, y_train, gvc__cat_features=cat_features, gvc__eval_set=(X_test, y_test), gvc__use_best_model=True)
    y_pred = model.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    scores.append(score)
    print("Balanced Accuracy:", score)
print("Mean Balanced Accuracy:", np.mean(scores), '+-', np.std(scores))

# %% group voting bayes


scores = []
X_data = X
Y_data = Y_

sgkfs = StratifiedGroupKFoldStrict(n_splits=5, shuffle=True, random_state=42)

opt = BayesSearchCV(
estimator=model,
search_spaces=search_space,
n_iter=5,                                # start with ~40-80; increase if time allows
scoring=make_scorer(balanced_accuracy_score),
cv=sgkfs,
n_jobs=-1,
refit=True,
random_state=42,
verbose=2
)
opt.fit(X_data, Y_data, groups=groups_all, cat_features=cat_features)

best_i = opt.best_index_
# %%
print(scores)
print("Mean Balanced Accuracy:", np.mean(scores), '+-', np.std(scores))
# %%
