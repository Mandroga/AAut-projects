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



# %% training data
#remove_feat_cols = ['Patient_Id', 'target','impairment_side']
remove_feat_cols = ['Patient_Id', 'target']
feat_cols = [col for col in df_processed.columns if col not in remove_feat_cols]
X_ = df_processed[feat_cols]
y_ = df_processed['target']
#w = np.array([1]*len(y_))

#if 1: Seperate test patients | else: regular split
if 1:
    Patient_Ids = list(range(1,15))
    Test_Ids = random.sample(Patient_Ids,4)
    mask = df_processed['Patient_Id'].isin(Train_Ids)
    X_train = X_[mask]
    y_train = y_[mask]
    w_train = w[mask]

    X_test = X_[~mask]
    y_test = y_[~mask]
    w_test = w[~mask]   
else:
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_, y_, w, test_size=0.2, stratify=y_, random_state=42)

# %% tts

#remove_feat_cols = ['Patient_Id', 'target','impairment_side']
remove_feat_cols = ['Patient_Id', 'target']
feat_cols = [col for col in df_processed.columns if col not in remove_feat_cols]
X_ = df_processed[feat_cols]
y_ = df_processed['target']
#w = np.array([1]*len(y_))

#if 1: Seperate test patients | else: regular split
if 1:
    Patient_Ids = list(range(1,15))
    Test_Ids = random.sample(Patient_Ids,4)
    mask = df_processed['Patient_Id'].isin(Train_Ids)
    X_train = X_[mask]
    y_train = y_[mask]
    w_train = w[mask]

    X_test = X_[~mask]
    y_test = y_[~mask]
    w_test = w[~mask]   
else:
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_, y_, w, test_size=0.2, stratify=y_, random_state=42)


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

# %% cv
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
# %% best model
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


# %% score

print(X_test.columns)
y_pred = clf.predict(X_test)
score = f1_score(y_test, y_pred, average='macro', sample_weight=w_test)
classification_r = classification_report(y_test, y_pred, sample_weight=w_test)
print(classification_r)

# %% confusion matrix
cm = confusion_matrix(y_test, y_pred)

# If your labels aren't 0..N-1, you can specify label names here:
labels = sorted(y_.unique())  # or use a custom list like ["class1", "class2", ...]

# Create display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Plot with better aesthetics
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap="Blues", values_format=".0f", ax=ax, colorbar=True)

plt.title("Confusion Matrix - XGBoost Model", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.show()

# %%
