# %% imports
%run data_imports.py

# %% thinking...
'''
Exercise recognition

O paciente esta sentado

E1 - Brushing hair - Passar a mao na cabeça - mãos 
E2 - Brushing teeth - Passar a mao na boca - mãos 
E5 - Hip flexion - Levanta os joelhos ao peito

O AVC afeta o paciente num dos lados =>
E1, E2 - Pode ou nao conseguir levantar a mao
E3 - Levanta sempre pelo menos um dos joelhos => O desvio padrao dos joelhos é maior que os outros exercicios

Ideia:
- Comparar a altura e o desvio padrao do que deve estar a mexer, com keypoints perto da zona de interesse~
(lavar os dentes - mao, nariz)
Logistic classification
Linear 
SVM
Forest classifier
Neural classifiers
'''

# %% df_
df_ = df.copy()

x_mean_i = [i for i in range(0, 66, 2)]
y_mean_i = [i for i in range(1, 66, 2)]
x_std_i = [i for i in range(66, 132, 2)]
y_std_i = [i for i in range(67, 132, 2)]
keypoints = list(range(33))
for i in keypoints:
    df_[f'xmean{i}'] = df_['Skeleton_Features'].apply(lambda x: x[x_mean_i[i]])
    df_[f'ymean{i}'] = df_['Skeleton_Features'].apply(lambda x: x[y_mean_i[i]])
    df_[f'xstd{i}'] = df_['Skeleton_Features'].apply(lambda x: x[x_std_i[i]])
    df_[f'ystd{i}'] = df_['Skeleton_Features'].apply(lambda x: x[y_std_i[i]])
df_ = df_.drop(['Skeleton_Features'], axis=1)
# %% preprocessing


def f1_macro_loss(y_true, y_pred, sample_weight):
    # y_pred is probs for multi:softprob; turn into labels
    y_pred = np.asarray(y_pred)
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)
    # return a *loss* (lower is better) since older XGB minimizes eval_metric
    return 1.0 - f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight)


indexes = [0, 19,20, 15, 16, 21, 22, 25,26, 31, 32]
#indexes = list(range(33))
keypoint_cols = [txt+str(i) for i in indexes for txt in ['xmean','ymean','xstd','ystd']]
cols = keypoint_cols + ['Patient_Id']

X_ = df_[keypoint_cols]
y_ = df_['target']

w = df_[[txt+str(j) for txt in ['xstd','ystd'] for j in [15,16,19,20,21,22]]]
#stdsc = StandardScaler()
scaler = MinMaxScaler()
w = scaler.fit_transform(w.values)
w =  w.sum(axis=1)
w *= w
#sns.histplot(w)
#plt.show()

# %% train test


X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_, y_, w, test_size=0.2, stratify=y_, random_state=42
)

# Classifier
clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=y_.nunique(),
    eval_metric=f1_macro_loss,
    tree_method='hist',
    early_stopping_rounds=50,
    learning_rate=0.05,
    n_estimators=500,
    max_depth=33,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

clf.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_test, y_test)],
    sample_weight_eval_set=[w_test],
    verbose=False
)
# %% score
y_pred = clf.predict(X_test)
score = f1_score(y_test, y_pred, average='macro', sample_weight=w_test)
classification_r = classification_report(y_test, y_pred, sample_weight=w_test)
print(classification_r)

# %% cv
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_, y_, w, test_size=0.2, stratify=y_, random_state=42
)

# ---------------------------
# 1) Base estimator (no early stopping in CV)
# ---------------------------
base_clf = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=int(y_.nunique()),
    tree_method="hist",     # switch to "gpu_hist" if you have a CUDA GPU
    random_state=42,
    eval_metric=f1_macro_loss  # CV uses macro-F1 via scoring; leave booster metric here
)

# ---------------------------
# 2) Search space + CV
# ---------------------------~
n_iter = 10
n_splits = 2
cv_inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
scorer = make_scorer(f1_score, average="macro")  # model selection metric

search_spaces = {
    "n_estimators": Integer(300, 2000),
    "max_depth": Integer(3, 12),
    "learning_rate": Real(1e-3, 3e-1, prior="log-uniform"),
    "subsample": Real(0.5, 1.0),
    "colsample_bytree": Real(0.5, 1.0),
    "min_child_weight": Real(1e-2, 10.0, prior="log-uniform"),
    "gamma": Real(0.0, 5.0),
    "reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),
    "reg_lambda": Real(1e-6, 10.0, prior="log-uniform"),
}

opt = BayesSearchCV(
    estimator=base_clf,
    search_spaces=search_spaces,
    n_iter=n_iter,                 # bump to 60–100 if you can
    cv=cv_inner,
    scoring=scorer,
    n_jobs=-1,
    refit=True,
    random_state=42,
    verbose=2,
)

# CV fit (pass weights)
opt.fit(X_train, y_train, sample_weight=w_train)

print("Best params:", opt.best_params_)

# ---------------------------
# 3) Final refit with early stopping (no CV leakage)
# ---------------------------
# %%
X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
    X_train, y_train, w_train, test_size=0.15, stratify=y_train, random_state=42
)

# start from the best params, give a large cap on trees, let early stopping pick the best
final_clf = xgb.XGBClassifier(
    **opt.best_params_,
    objective="multi:softprob",
    num_class=int(y_.nunique()),
    tree_method="hist",
    random_state=42,
    eval_metric='mlogloss',  # use a standard metric for early stopping
    early_stopping_rounds=50,
)

final_clf.fit(
    X_tr, y_tr,
    sample_weight=w_tr,
    eval_set=[(X_val, y_val)],
)

# ---------------------------
# 4) Evaluate on held-out test
# ---------------------------
y_pred = final_clf.predict(X_test)
print("Test F1-macro (unweighted):", f1_score(y_test, y_pred, average="macro"))
print("Test F1-macro (weighted by w_test):", f1_score(y_test, y_pred, average="macro", sample_weight=w_test))
print(classification_report(y_test, y_pred, digits=3))
# %%
