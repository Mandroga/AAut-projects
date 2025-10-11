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



# %% training data df

df['weights'] = w
n1 = 12
n2 = 4
n3= 1

df_train = df[df["Patient_Id"] != n1]
df_train = df_train[df_train["Patient_Id"] != n2]
df_train = df_train[df_train["Patient_Id"] != n3]

X_train = np.array(df_train['Skeleton_Features'].to_list())
y_train = df_train['target']
w_train = df_train['weights']

df_test = df[df["Patient_Id"].isin([n1, n2])]

X_test = np.array(df_test['Skeleton_Features'].to_list())


y_test = df_test['target']


n_iter = 10

# %% Grouped CV
clf = xgb.XGBClassifier(
    objective="multi:softprob",
    random_state=42,
    eval_metric='mlogloss',  # CV uses macro-F1 via scoring; leave booster metric here
)

model = Pipeline([('features',FeatureTransform_np()),('clf', clf)])

# ---------------------------
# 2) Search space + CV
# ---------------------------~
scorer = make_scorer(f1_score, average="macro")  # model selection metric

search_spaces = {
    "clf__n_estimators": Integer(300, 2000),
    "clf__max_depth": Integer(3, 12),
    "clf__learning_rate": Real(1e-3, 3e-1, prior="log-uniform"),
    "clf__subsample": Real(0.5, 1.0),
    "clf__colsample_bytree": Real(0.5, 1.0),
    "clf__min_child_weight": Real(1e-2, 10.0, prior="log-uniform"),
    "clf__gamma": Real(0.0, 5.0),
    "clf__reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),
    "clf__reg_lambda": Real(1e-6, 10.0, prior="log-uniform"),
}

opt = BayesSearchCV(
    estimator=model,
    search_spaces=search_spaces,
    n_iter=n_iter,                 # bump to 60–100 if you can
    cv=5,
    scoring=scorer,
    n_jobs=-1,
    refit=True,
    random_state=42,
    verbose=2,
)

# CV fit (pass weights)
opt.fit(X_train, y_train, clf__sample_weight=w_train)
y_pred = opt.predict(X_test)

print("Best params:", opt.best_params_)
print("Test F1-macro (unweighted):", f1_score(y_test, y_pred, average="macro"))
print(classification_report(y_test, y_pred, digits=3))
# %% score

clf.fit(X_train, y_train, sample_weight=w_train)
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
