# %% imports
from imports3 import *

# %% load data
with open("Xtrain2.pkl", "rb") as f:
    X = pickle.load(f)
Y= np.load("Ytrain2.npy")
# %% preprocess
print(X.columns)
class preprocess_data(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.all_keypoints = {'r':[4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
            'l':[1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31]}
        self.body_parts = {'leg':{'l':[23,25,27,29,31],'r':[24,26,28,30,32]},
            'arm':{'l':[11,13,15,17,19,21],'r':[12,14,16,18,20,22]},
            'torso':{'l':[11,23],'r':[12,24]},
            'face':{'l':[1,2,3,7,9],'r':[4,5,6,8,10]}}
        
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        for i in range(len(X)):
            patient_id = X.loc[i,'Patient_Id']
            exercise_id = X.loc[i,'Exercise_Id']
            X_ss = X.loc[i,'Skeleton_Sequence']
            X_ss_df = skeleton_sequence_to_df(X_ss)
            X_ss_df = df_distances(X_ss_df, range(33))
            for body_part in self.body_parts.keys():
                for key in ['l', 'r']:
                    dist_cols = make_cols(self.body_parts[body_part][key], ['dist'])
                    total_distances = X_ss_df[dist_cols].sum(axis=1)
                    X.loc[i,f'{body_part}_{key}_distance_mean'] = total_distances.mean()
                    X.loc[i,f'{body_part}_{key}_distance_std'] = total_distances.std()
                    X.loc[i,f'{body_part}_{key}_distance_median'] = total_distances.median()
        if 1:
            if exercise_id == "E3":
                lpinky = X_ss[:,17*2:17*2+1+1]
                rpinky = X_ss[:,18*2:18*2+1+1]
                lindex = X_ss[:,19*2:19*2+1+1]
                rindex = X_ss[:,20*2:20*2+1+1]
                dis_pink_index_l = np.hypot(lpinky[:,0]-lindex[:,0], lpinky[:,1]-lindex[:,1]).mean()
                dis_pink_index_r = np.hypot(rpinky[:,0]-rindex[:,0], rpinky[:,1]-rindex[:,1]).mean()
                X.loc[i,'dis_pink_index_l'] = dis_pink_index_l
                X.loc[i,'dis_pink_index_r'] = dis_pink_index_r


                #X.loc[i,'amplitude_l_index'] = amplitude_l_index
                #X.loc[i,'amplitude_r_index'] = amplitude_r_index
            else:
                X.loc[i,'dis_pink_index_l'] = np.nan
                X.loc[i,'dis_pink_index_r'] = np.nan
                X.loc[i,'norm_amplitude_l_index'] = np.nan
                X.loc[i,'norm_amplitude_r_index'] = np.nan
                #X.loc[i,'amplitude_l_index'] = np.nan
                #X.loc[i,'amplitude_r_index'] = np.nan
            if exercise_id == "E4":
                lindex = X_ss[:,19*2:19*2+1+1]
                rindex = X_ss[:,20*2:20*2+1+1]
                
                #Amplitudes and distances for index fingers
                amplitude_l_index = lindex[:,1].max() - lindex[:,1].min()
                amplitude_r_index = rindex[:,1].max() - rindex[:,1].min()

                distance_l_index = sum_consecutive_distances(lindex)
                distance_r_index = sum_consecutive_distances(rindex)

                norm_amplitude_l_index =  2 * amplitude_l_index / distance_l_index if distance_l_index !=0 else 0
                norm_amplitude_r_index = 2 * amplitude_r_index / distance_r_index if distance_r_index !=0 else 0
                X.loc[i,'norm_amplitude_l_index'] = norm_amplitude_l_index
                X.loc[i,'norm_amplitude_r_index'] = norm_amplitude_r_index
                X.loc[i,'amplitude_l_index'] = amplitude_l_index
                X.loc[i,'amplitude_r_index'] = amplitude_r_index

                #Orientation changes in wrist keypoints
                lwrist = X_ss[:,17*2:17*2+1+1]
                rwrist = X_ss[:,16*2:16*2+1+1]
                orientation_changes_lwrist = orientationchange(lwrist)
                orientation_changes_rwrist = orientationchange(rwrist)
                #X.loc[i,'orientation_changes_lwrist'] = orientation_changes_lwrist
                #X.loc[i,'orientation_changes_rwrist'] = orientation_changes_rwrist

            else:
                X.loc[i,'norm_amplitude_l_index'] = np.nan
                X.loc[i,'norm_amplitude_r_index'] = np.nan
                X.loc[i,'amplitude_l_index'] = np.nan
                X.loc[i,'amplitude_r_index'] = np.nan
                #X.loc[i,'orientation_changes_lwrist'] = np.nan
                #X.loc[i,'orientation_changes_rwrist'] = np.nan

            if 1:
                if exercise_id == "E5":
                    lknee = X_ss[:,25*2:25*2+1+1]
                    rknee = X_ss[:,26*2:26*2+1+1]
                    orientation_changes_lknee = orientationchange(lknee)
                    orientation_changes_rknee = orientationchange(rknee)
                    X.loc[i,'orientation_changes_lknee'] = orientation_changes_lknee
                    X.loc[i,'orientation_changes_rknee'] = orientation_changes_rknee

                    #Amplitudes and distances for knee keypoints
                    amplitude_l_knee = lknee[:,1].max() - lknee[:,1].min()
                    amplitude_r_knee = rknee[:,1].max() - rknee[:,1].min()

                    distance_l_knee = sum_consecutive_distances(lknee)
                    distance_r_knee = sum_consecutive_distances(rknee)

                    norm_amplitude_l_knee =  2 * amplitude_l_knee / distance_l_knee if distance_l_knee !=0 else 0
                    norm_amplitude_r_knee = 2 * amplitude_r_knee / distance_r_knee if distance_r_knee !=0 else 0

                    X.loc[i,'norm_amplitude_l_knee'] = norm_amplitude_l_knee
                    X.loc[i,'norm_amplitude_r_knee'] = norm_amplitude_r_knee
                    X.loc[i,'amplitude_l_knee'] = amplitude_l_knee
                    X.loc[i,'amplitude_r_knee'] = amplitude_r_knee
                else:
                    X.loc[i,'orientation_changes_lknee'] = np.nan
                    X.loc[i,'orientation_changes_rknee'] = np.nan
                    X.loc[i,'norm_amplitude_l_knee'] = np.nan
                    X.loc[i,'norm_amplitude_r_knee'] = np.nan
                    X.loc[i,'amplitude_l_knee'] = np.nan
                    X.loc[i,'amplitude_r_knee'] = np.nan
            
        
        
        #drop ss
        if 1:
            X = X.drop('Skeleton_Sequence', axis=1)
        return X

active_classes = np.where(Y==1)[0]+1
Y_ = X['Patient_Id'].isin(active_classes).astype(int)
X_pre = preprocess_data().fit_transform(X)
groups_all = X['Patient_Id'].to_numpy()

print(X_pre)

# %% classifier
model = CatBoostClassifier(
    iterations=100,        # número de árvores
    learning_rate=0.05,    # taxa de aprendizado
    depth=5,               # profundidade das árvores
    loss_function='Logloss',
    eval_metric='BalancedAccuracy',
    random_seed=42,
    verbose=100            # mostra progresso a cada 100 iterações
)
cat_features = ['Patient_Id','Exercise_Id']

search_space = {
    "iterations": (200, 1200),                 # int
    "learning_rate": (1e-3, 0.2, "log-uniform"),
    "depth": (4, 10),                          # int
    "l2_leaf_reg": (1e-2, 20.0, "log-uniform"),
    "min_data_in_leaf": (1, 64),               # int
    "bagging_temperature": (0.0, 1.0),         # only used when bootstrap_type="Bayesian"
    "subsample": (0.5, 1.0),                   # only used when bootstrap_type="Bernoulli"
    "colsample_bylevel": (0.5, 1.0),
    "random_strength": (0.0, 2.0),
    "bootstrap_type": ["Bayesian", "Bernoulli"],
    "grow_policy": ["SymmetricTree"],          # (Lossguide) is OK too; add if you want
}


# %%

# %% training no tuning

scores = []
X_data = X_pre
Y_data = Y_

sgkfs = StratifiedGroupKFoldStrict(n_splits=3, shuffle=True, random_state=42)

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
print("Mean Balanced Accuracy:", np.mean(scores), '+-', np.std(scores))

# %% training bayes

scores = []
X_data = X_pre
Y_data = Y_

sgkfs = StratifiedGroupKFoldStrict(n_splits=3, shuffle=True, random_state=42)

opt = BayesSearchCV(
estimator=model,
search_spaces=search_space,
n_iter=5,                                # start with ~40-80; increase if time allows
scoring=make_scorer(balanced_accuracy_score),
cv=sgkfs,
n_jobs=-1,
refit=True,
random_state=42
)
opt.fit(X_data, Y_data, groups=groups_all, cat_features=cat_features)

best_i = opt.best_index_

# Grab all split columns
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

# %%