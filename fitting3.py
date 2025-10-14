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
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        for i in range(len(X)):
            patient_id = X.loc[i,'Patient_Id']
            exercise_id = X.loc[i,'Exercise_Id']
            X_ss = X.loc[i,'Skeleton_Sequence']
            X_ss_df = skeleton_sequence_to_df(X_ss)
            X_ss_df = df_distances(X_ss_df, range(33))
            for key in ['l', 'r']:
                dist_cols = make_cols(self.all_keypoints[key], ['dist'])
                total_distances = X_ss_df[dist_cols].sum().sum() / len(X_ss_df)
                X.loc[i,f'total_{key}_distances'] = total_distances
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
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
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


# %%

# %% training
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, balanced_accuracy_score


scores = []
sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
X_data = X_pre
Y_data = Y_
for iteration, (train_val_idx, test_idx) in enumerate(sgkf.split(X_data, Y_data, groups=groups_all)):
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
print("Mean Balanced Accuracy:", np.mean(scores))
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
