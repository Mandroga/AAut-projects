# %%
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
indexes = [0, 19,20, 15, 16, 21, 22, 25,26, 31, 32]
#indexes = list(range(33))
keypoint_cols = [txt+str(i) for i in indexes for txt in ['xmean','ymean','xstd','ystd']]
cols = keypoint_cols + ['Patient_Id']

w = df_[[txt+str(j) for txt in ['xstd','ystd'] for j in [15,16,19,20, 21, 22]]]
#stdsc = StandardScaler()
scaler = MinMaxScaler()
w = scaler.fit_transform(w.values)
w =  w.sum(axis=1)
sns.histplot(w)
plt.show()

# %%
# Split
X_ = df_[keypoint_cols]
y_ = df_['target']
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_, y_, w, test_size=0.2, stratify=y_, random_state=42
)

# Classifier
clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=y_.nunique(),
    eval_metric='mlogloss',
    tree_method='hist',
    learning_rate=0.05,
    n_estimators=2000,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# Fit with sample weights (and weight the eval set too)

es = xgb.callback.EarlyStopping(
    rounds=50,     # patience
    save_best=True,
    maximize=False # True if your metric should be maximized
)

clf.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_test, y_test)],
    sample_weight_eval_set=[w_test],
   # callbacks=[es],
    verbose=False
)
# %%
y_pred = clf.predict(X_test)
score = f1_score(y_test, y_pred, average='macro', sample_weight=w_test)
print(score)

# %%
