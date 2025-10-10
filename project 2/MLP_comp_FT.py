#%%



%run data_imports.py


#%%

import pickle

from data_imports import FeatureTransform_np

path = "Xtrain1.pkl"
with open(path, "rb") as f:
    X_df = pickle.load(f)

X = np.stack(X_df["Skeleton_Features"].values)
Y = np.load("Ytrain1.npy")
#%%
X = df["Skeleton_Features"].values   # or df.drop(columns=["label", "patient_id"]) if using tabular features
y = df["target"].values
groups = df["Patient_Id"].values

from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]


#%%
n1 = 12
n2 = 4

df_train = df[df["Patient_Id"] != n1]
df_train = df_train[df_train["Patient_Id"] != n2]

X_train = np.array(df_train['Skeleton_Features'].to_list())


df_test = df[df["Patient_Id"].isin([n1, n2])]

X_test = np.array(df_test['Skeleton_Features'].to_list())

y_train = df_train['target']
y_test = df_test['target']


#%% separate test train


class PreProcessing(BaseEstimator, TransformerMixin):
    """
    Custom preprocessing transformer for scikit-learn pipelines.
    Implements fit/transform methods so it can be used in a Pipeline.
    """

    def __init__(self):
        """
        Initialize any hyperparameters here.
        They will be stored as attributes.
        Example: param1 = scaling factor, param2 = threshold, etc.
        """
        self.l_arm_arr = np.array([])
        self.r_arm_arr= np.array([])
        self.l_leg_arr=np.array([])
        self.r_leg_arr=np.array([])
        self.heights =np.array([])
        self.heights_d =np.array([])
        self.widths=np.array([])
        self.widths_d= np.array([])
        self.l_hny=np.array([])
        self.r_hny=np.array([])
        self.d_l_foot_hand = np.array([])
        self.d_r_foot_hand = np.array([])

    def fit(self, X, y=None):
        """
        Learn any statistics from the data, if needed.
        For example: compute mean/std for scaling, or label encoder mapping.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
        y : array-like, optional, shape (n_samples,)
        
        Returns:
        self : object
        """


        return self
    
    @staticmethod
    def update_skeleton_features(feat_arr):
        feat_arr = list(feat_arr)

        featkeep = [15,16,17,18,19,20,21,22,27,28]
        feat_arr = [feat_arr[i*2] for i in featkeep]+[feat_arr[i*2+1]for i in featkeep]+[feat_arr[i*4] for i in featkeep]+[feat_arr[i*4+1]for i in featkeep]


        return feat_arr
    

    def transform(self, X):
        """
        Apply the preprocessing to X.
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        X_transformed : array-like, shape (n_samples, n_features_new)
        """

        # Distance from shoulder to hand
        l_shoulder_pos = X[:, 11*2:11*2+2]  # shape (n_samples, 2)
        r_shoulder_pos = X[:, 12*2:12*2+2]
        l_hand_pos = X[:, 19*2:19*2+2]
        r_hand_pos = X[:, 20*2:20*2+2]

        self.l_arm_arr = np.linalg.norm(l_hand_pos - l_shoulder_pos, axis=1)
        self.r_arm_arr = np.linalg.norm(r_hand_pos - r_shoulder_pos, axis=1)

        # Height
        l_eye_pos = X[:, 3*2:3*2+2]
        r_eye_pos = X[:, 6*2:6*2+2]
        l_foot_pos = X[:, 29*2:29*2+2]
        r_foot_pos = X[:, 30*2:30*2+2]

        l_height = np.linalg.norm(l_eye_pos - l_foot_pos, axis=1)
        r_height = np.linalg.norm(r_eye_pos - r_foot_pos, axis=1)
        self.heights_d = np.maximum(l_height, r_height)

        l_height_y = l_eye_pos[:,1] - l_foot_pos[:,1]
        r_height_y = r_eye_pos[:,1] - r_foot_pos[:,1]
        self.heights = np.maximum(l_height_y, r_height_y)

        # Width (hip distance)
        l_hip_pos = X[:, 23*2:23*2+2]
        r_hip_pos = X[:, 24*2:24*2+2]
        self.widths_d = np.linalg.norm(l_hip_pos - r_hip_pos, axis=1)

        l_hip_x_pos = X[:, 23*2]
        r_hip_x_pos = X[:, 24*2]
        self.widths = l_hip_x_pos-r_hip_x_pos

        # Distance from foot to hip
        self.l_leg_arr = np.linalg.norm(l_hip_pos - l_foot_pos, axis=1)
        self.r_leg_arr = np.linalg.norm(r_hip_pos - r_foot_pos, axis=1)

        # Hand - nose difference in y
        self.l_hny = X[:,0*2+1] - X[:,19*2+1]
        self.r_hny = X[:,0*2+1] - X[:,20*2+1]
        

        normalized_r_arm = self.r_arm_arr / self.heights
        normalized_l_arm = self.l_arm_arr / self.heights

        normalized_r_leg=self.r_leg_arr /self.heights
        normalized_l_leg=self.l_leg_arr / self.heights

        normalized_r_hand_std = X[:,20*4+1]/self.widths
        normalized_l_hand_std = X[:,19*4+1]/self.widths

        #Elbow and knee stdv
        featkeep = [13,14,25,26]
        elbow_knee_sd = [X[:,i*4] / self.widths for i in featkeep]+[X[:,i*4+1] /self.heights for i in featkeep]

        elbow_knee_sd = np.array(elbow_knee_sd).T

        # Height
        l_eye_pos = X[:, 3*2:3*2+2]
        r_eye_pos = X[:, 6*2:6*2+2]
        l_toe_pos = X[:, 29*2:29*2+2]
        r_toe_pos = X[:, 30*2:30*2+2]

        l_height = np.linalg.norm(l_eye_pos - l_toe_pos, axis=1)
        r_height = np.linalg.norm(r_eye_pos - r_toe_pos, axis=1)
        self.heights_d = np.maximum(l_height, r_height)

        l_height_y = l_eye_pos[:,1] - l_toe_pos[:,1]
        r_height_y = r_eye_pos[:,1] - r_toe_pos[:,1]
        self.heights = np.maximum(l_height_y, r_height_y)

        # Width (hip distance)
        l_hip_pos = X[:, 23*2:23*2+2]
        r_hip_pos = X[:, 24*2:24*2+2]
        self.widths_d = np.linalg.norm(l_hip_pos - r_hip_pos, axis=1)

        l_hip_x_pos = X[:, 23*2]
        r_hip_x_pos = X[:, 24*2]
        self.widths = l_hip_x_pos-r_hip_x_pos

        r_hand = [16,18,20,22]
        l_hand=[15,17,19,21]

        r_foot=[28,30,32]
        l_foot=[27,29,31]

        # Standard deviation feet and hands
        r_foot_x_sd = X[:, [i*4 for i in r_foot]].mean(axis=1) / self.widths
        r_foot_y_sd = X[:, [i*4 + 1 for i in r_foot]].mean(axis=1) / self.heights

        l_foot_x_sd = X[:, [i*4 for i in l_foot]].mean(axis=1) / self.widths
        l_foot_y_sd = X[:, [i*4 + 1 for i in l_foot]].mean(axis=1) / self.heights

        r_hand_x_sd = X[:, [i*4 for i in r_hand]].mean(axis=1) / self.widths
        r_hand_y_sd = X[:, [i*4 + 1 for i in r_hand]].mean(axis=1) / self.heights

        l_hand_x_sd = X[:, [i*4 for i in l_hand]].mean(axis=1) / self.widths
        l_hand_y_sd = X[:, [i*4 + 1 for i in l_hand]].mean(axis=1) / self.heights


        # Foot and hand average pos
        
        r_foot_x = X[:, [i*2 for i in r_foot]].mean(axis=1) 
        r_foot_y = X[:, [i*2 + 1 for i in r_foot]].mean(axis=1)

        l_foot_x = X[:, [i*2 for i in l_foot]].mean(axis=1) 
        l_foot_y = X[:, [i*2 + 1 for i in l_foot]].mean(axis=1)

        r_hand_x = X[:, [i*2 for i in r_hand]].mean(axis=1) 
        r_hand_y = X[:, [i*2 + 1 for i in r_hand]].mean(axis=1)

        l_hand_x = X[:, [i*2 for i in l_hand]].mean(axis=1) 
        l_hand_y = X[:, [i*2 + 1 for i in l_hand]].mean(axis=1)

        # Distance hand feet
        r_foot_pos = np.column_stack((r_foot_x, r_foot_y))
        l_foot_pos = np.column_stack((l_foot_x, l_foot_y))
        r_hand_pos = np.column_stack((r_hand_x, r_hand_y))
        l_hand_pos = np.column_stack((l_hand_x, l_hand_y))

        self.d_r_foot_hand = np.linalg.norm(r_foot_pos-r_hand_pos, axis=1)
        self.d_l_foot_hand = np.linalg.norm(l_foot_pos-l_hand_pos, axis=1)
        
        normalized_d_r_foot_hand = self.d_r_foot_hand /self.heights
        normalized_d_l_foot_hand = self.d_l_foot_hand / self.heights


        #x and y difference hand feet
        r_foot_hand_x = (r_foot_x-r_hand_x) /self.widths
        r_foot_hand_y = (r_foot_y-r_hand_y) /self.heights
        l_foot_hand_x = (l_foot_x-l_hand_x) /self.widths
        l_foot_hand_y = (l_foot_y-l_hand_y) / self.heights
        

        # Normalized feet and hand pos
        r_foot_x_normalized = X[:, [i*2 for i in r_foot]].mean(axis=1) / self.widths
        r_foot_y_normalized = X[:, [i*2 + 1 for i in r_foot]].mean(axis=1) / self.heights

        l_foot_x_normalized = X[:, [i*2 for i in l_foot]].mean(axis=1) / self.widths
        l_foot_y_normalized = X[:, [i*2 + 1 for i in l_foot]].mean(axis=1) / self.heights

        r_hand_x_normalized = X[:, [i*2 for i in r_hand]].mean(axis=1) / self.widths
        r_hand_y_normalized = X[:, [i*2 + 1 for i in r_hand]].mean(axis=1) / self.heights

        l_hand_x_normalized = X[:, [i*2 for i in l_hand]].mean(axis=1) / self.widths
        l_hand_y_normalized = X[:, [i*2 + 1 for i in l_hand]].mean(axis=1) / self.heights
        
        # Update skeleton features row-wise
        X_transformed_list = [self.update_skeleton_features(row) for row in X]

        # Convert to 2D NumPy array
        X_transformed_array = np.array(X_transformed_list)

        # Stack extra normalized features
        extra_features = np.column_stack([
            normalized_l_arm,
            normalized_r_arm,
            normalized_l_leg,
            normalized_r_leg,
            normalized_l_hand_std,
            normalized_r_hand_std,
            self.l_hny,
            self.r_hny,
            elbow_knee_sd,    
            r_foot_x_sd,
            r_foot_y_sd,
            l_foot_x_sd,
            l_foot_y_sd,
            r_hand_x_sd,
            r_hand_y_sd,
            l_hand_x_sd,
            l_hand_y_sd,
            r_foot_x_normalized,
            r_foot_y_normalized,
            l_foot_x_normalized,
            l_foot_y_normalized,
            r_hand_x_normalized,
            r_hand_y_normalized,
            l_hand_x_normalized,
            l_hand_y_normalized,   
        ])
        # Combine original features with normalized features
        X_transformed = np.hstack([X, extra_features])

        return X_transformed


# %%
#X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=RANDOM_STATE, stratify=Y)

#%%

pipe = Pipeline([
    ('my_pre_processing', PreProcessing()),
    ('features',FeatureTransform_np()),
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=500, random_state=RANDOM_STATE, early_stopping=True,validation_fraction=0.1))
])

param_grid = {
    'mlp__hidden_layer_sizes': [(100,), (100,50), (100,100,100)],
    'mlp__activation': ['relu', 'tanh', "logistic"],
    'mlp__solver': ['adam', 'sgd'],
    'mlp__alpha': [1e-4, 1e-3],
    'mlp__learning_rate_init': [1e-3, 1e-4],
    'mlp__learning_rate': ['constant', 'adaptive'],
    'mlp__n_iter_no_change': [6, 10, 20],
    'mlp__tol': [1e-4, 1e-5]

}

grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='f1_macro')
grid.fit(X_train, y_train)
#%%
print("Best params MLP:", grid.best_params_)
print("Best score MLP: ", grid.best_score_)
y_pred = grid.predict(X_test)
print("n1, n2 =", n1, n2, classification_report(y_test, y_pred))
# %%
from sklearn.svm import SVC


pipe_SVM = Pipeline([
    ('my_pre_processing', PreProcessing()),
    ('features',FeatureTransform_np()),
    ('scaler', StandardScaler()),
    ('svm', SVC())
    ])

param_grid_SVM = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1],
    'svm__degree': [2, 3],
    'svm__class_weight': [None, 'balanced']
}

grid_SVM = GridSearchCV(pipe_SVM, param_grid_SVM, cv=5, n_jobs=-1, verbose=1, scoring='f1_macro')
grid_SVM.fit(X_train, y_train)
#%%
print("Best params SVM:", grid_SVM.best_params_)
print("Best score SVM: ", grid_SVM.best_score_)
y_pred = grid_SVM.predict(X_test)
print("n1, n2 =", n1, n2)
print(classification_report(y_test, y_pred))

# %%