# %% imports
%run imports.py

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



# %% load data

with open("Xtrain1.pkl", "rb") as f:
    X = pickle.load(f)
Y= np.load("Ytrain1.npy")

groups_all = X['Patient_Id'].to_numpy()
X_np = np.array(X['Skeleton_Features'].to_list())

# %% classifier and scorer
#xgb
if 0:
    clf_name = 'xgb'
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        random_state=42,
        eval_metric='mlogloss',  # CV uses macro-F1 via scoring; leave booster metric here
    )

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
#mlp
if 0:
    clf_name = 'mlp'
    clf = MLPClassifier(
    hidden_layer_sizes=(132,),   # <-- inside your space
    activation='tanh',
    solver='adam',
    alpha=1e-4,
    learning_rate_init=1e-3,
    early_stopping=True,
    n_iter_no_change=50,
    max_iter=2000,
    random_state=42
    )
    
    search_spaces = {
   # 'clf__activation': Categorical(['relu', 'tanh', 'logistic']),
    'clf__alpha': Real(1e-6, 1e-2, prior='log-uniform'),
    'clf__learning_rate_init': Real(1e-4, 1e-1, prior='log-uniform'),
    'clf__beta_1': Real(0.8, 0.999),
    'clf__beta_2': Real(0.9, 0.9999),
    'clf__epsilon': Real(1e-9, 1e-7, prior='log-uniform'),
}

#svm rbf
if 1:
    clf_name = 'svm_rbf'

    # from skopt.space import Real, Integer, Categorical  # assuming you already import these

    # ----- pipeline -----
    clf = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',          # good default; will be tuned below
        tol=1e-3,
        shrinking=True,
        probability=False,      # set True if you need predict_proba
        cache_size=500,
        class_weight=None,
        random_state=42
    )


    # ----- search space -----
    search_spaces = {
        'clf__C': Real(1e-3, 1e+3, prior='log-uniform'),
        'clf__gamma': Real(1e-6, 1e0, prior='log-uniform'),
        'clf__tol': Real(1e-5, 1e-2, prior='log-uniform'),
        'clf__shrinking': Categorical([True, False]),
        'clf__class_weight': Categorical([None, 'balanced']),
    }

scorer = make_scorer(f1_score, average="macro")  # model selection metric
# %% nested sgkf
# ----- outer CV: leave-one-group-out (1 grupo de teste) -----

unique_groups = np.unique(groups_all)

if 0: # 2 for testing 2 for validation
    outer_split = 7#len(unique_groups)
    inner_split = 6

if 1: # 2 for testing 2 for validation
    outer_split = 14 #len(unique_groups)
    inner_split = 13

outer_sgkf = StratifiedGroupKFold(
    n_splits=outer_split, shuffle=True, random_state=42
   #n_splits=5, shuffle=True, random_state=42
)

outer_results = []
per_fold_reports = {}
best_params_per_fold = {}

fold_id = 0
for train_val_idx, test_idx in outer_sgkf.split(X_np, Y, groups=groups_all):
    print('outer split')
    fold_id += 1
    X_trv, X_te = X_np[train_val_idx], X_np[test_idx]
    y_trv, y_te = Y[train_val_idx], Y[test_idx]
    groups_trv = groups_all[train_val_idx]

    # ----- inner CV: leave-one-group-out nos grupos de treino (1 grupo de validação por fold) -----
    inner_sgkf = StratifiedGroupKFold(
        n_splits=inner_split, shuffle=True, random_state=42
    )

    # pipeline com transformer + scaler dentro do CV (evita leakage)
    pipe = Pipeline([
        ('features', FeatureTransform3()),
        ('scaler', StandardScaler()),
        ('clf', clf),
    ])

    opt = BayesSearchCV(
        estimator=pipe,
        search_spaces=search_spaces,
        n_iter=5,                 # aumenta se quiseres explorar mais
        cv=inner_sgkf,
        scoring=scorer,
        n_jobs=-1,
        refit=True,
        random_state=42,
        verbose=2,
    )

    # importante: passar groups no fit do inner CV
    opt.fit(X_trv, y_trv, groups=groups_trv)

    # melhor modelo do inner CV → avaliar no grupo de teste externo
    best_model = opt.best_estimator_
    y_pred = best_model.predict(X_te)

    f1_macro = f1_score(y_te, y_pred, average='macro')
    outer_results.append(f1_macro)
    best_params_per_fold[f'fold_{fold_id}'] = opt.best_params_

    # relatório por fold (opcional)
    per_fold_reports[f'fold_{fold_id}'] = (classification_report(y_te, y_pred, digits=3), np.unique(groups_trv))

    # salvar modelo por fold (opcional)
    #joblib.dump(best_model, f'{clf_name}_pipeline_fold{fold_id}.joblib')

# ---- resumo ----
outer_results = np.array(outer_results)

print(f'[Nested SGKF] {clf_name} | F1-macro por fold (teste externo): {np.round(outer_results, 4)}')
print(f'[Nested SGKF] Média ± DP: {outer_results.mean():.4f} ± {outer_results.std():.4f}')

pd.set_option('display.max_colwidth', None)   # não corta texto de colunas
pd.set_option('display.max_columns', None)    # mostra todas as colunas
pd.set_option('display.width', None)          # não quebra linha automaticamente
pd.set_option('display.max_rows', None)  

from io import StringIO
test_group_scores = []
reps_np = []
with open(f"reports\{clf_name}_reports_all.txt", "w") as f:
    f.write('\n\n\n')
    f.write(f'[Nested SGKF] {clf_name} | F1-macro por fold (teste externo): {np.round(outer_results, 4)}')
    f.write(f'[Nested SGKF] Media +- DP: {outer_results.mean():.4f} +- {outer_results.std():.4f}')
    for k, (rep, groups) in per_fold_reports.items():
        f.write(f"\n{k} | classification report (outer test):\n{rep}\n")
        f.write(f"Groups in training set: {groups}\n")
        test_groups = [i for i in range(1,15) if i not in groups]
        f.write(f"Groups in test set: {test_groups}\n")
        df = pd.read_fwf(StringIO(rep))
        
        rep_np = df[['precision','recall','f1-score']].to_numpy()
        reps_np.append(rep_np)
        test_group_scores.append((test_groups, float(df.loc[4, 'f1-score'])))
    
    reps_np = np.array(reps_np)
    reps_np_mean = np.mean(reps_np, axis=0)
    reps_np_std = np.std(reps_np, axis=0)
    reps_np_cols = np.concatenate([reps_np_mean, reps_np_std], axis=1)
    fold_average_report = pd.DataFrame(reps_np_cols, columns=['precision mean','recall mean','f1-score mean',
                                                             'precision std','recall std','f1-score std'])
    fold_average_report = fold_average_report[['precision mean','precision std','recall mean','recall std','f1-score mean','f1-score std']]
    fold_average_report['name'] = ['0','1','2','accuracy','macro avg','weighted avg']
    fold_average_report = fold_average_report.set_index('name')
    f.write(f"\n\nAverage classification report over folds:\n{fold_average_report}\n")
    f.write(f"\nTest group scores: {sorted(test_group_scores,key=lambda x: x[1])}\n")
# %% Results
print(f'[Nested SGKF] {clf_name} | F1-macro por fold (teste externo): {np.round(outer_results, 4)}')
print(f'[Nested SGKF] Média ± DP: {outer_results.mean():.4f} ± {outer_results.std():.4f}')

pd.set_option('display.max_colwidth', None)   # não corta texto de colunas
pd.set_option('display.max_columns', None)    # mostra todas as colunas
pd.set_option('display.width', None)          # não quebra linha automaticamente
pd.set_option('display.max_rows', None)  

from io import StringIO
test_group_scores = []
reps_np = []
with open(f"reports\{clf_name}_reports_all.txt", "w") as f:
    f.write('\n\n\n')
    f.write(f'[Nested SGKF] {clf_name} | F1-macro por fold (teste externo): {np.round(outer_results, 4)}')
    f.write(f'[Nested SGKF] Media +- DP: {outer_results.mean():.4f} +- {outer_results.std():.4f}')
    for k, (rep, groups) in per_fold_reports.items():
        f.write(f"\n{k} | classification report (outer test):\n{rep}\n")
        f.write(f"Groups in training set: {groups}\n")
        test_groups = [i for i in range(1,15) if i not in groups]
        f.write(f"Groups in test set: {test_groups}\n")
        df = pd.read_fwf(StringIO(rep))
        
        rep_np = df[['precision','recall','f1-score']].to_numpy()
        reps_np.append(rep_np)
        test_group_scores.append((test_groups, float(df.loc[4, 'f1-score'])))
    
    reps_np = np.array(reps_np)
    reps_np_mean = np.mean(reps_np, axis=0)
    reps_np_std = np.std(reps_np, axis=0)
    reps_np_cols = np.concatenate([reps_np_mean, reps_np_std], axis=1)
    fold_average_report = pd.DataFrame(reps_np_cols, columns=['precision mean','recall mean','f1-score mean',
                                                             'precision std','recall std','f1-score std'])
    fold_average_report = fold_average_report[['precision mean','precision std','recall mean','recall std','f1-score mean','f1-score std']]
    fold_average_report['name'] = ['0','1','2','accuracy','macro avg','weighted avg']
    fold_average_report = fold_average_report.set_index('name')
    f.write(f"\n\nAverage classification report over folds:\n{fold_average_report}\n")
    f.write(f"\nTest group scores: {sorted(test_group_scores,key=lambda x: x[1])}\n")
#%%
from io import StringIO
for k, (rep, groups) in per_fold_reports.items():  
    df = pd.read_fwf(StringIO(rep))
    print(df)

# %% confusion matrix
cm = confusion_matrix(y_test, y_pred)

# If your labels aren't 0..N-1, you can specify label names here:
labels = sorted(np.unique(Y))  # or use a custom list like ["class1", "class2", ...]

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


