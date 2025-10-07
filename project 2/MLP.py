
#%%

%run data_imports.py

#%%



X= np.array(df['Skeleton_Features'].to_list())
Y=np.load("Ytrain1.npy")
#%%    


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
        self.widths=np.array([])
        self.l_hny=np.array([])
        self.r_hny=np.array([])

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
        self.heights = np.maximum(l_height, r_height)

        # Width (hip distance)
        l_hip_pos = X[:, 23*2:23*2+2]
        r_hip_pos = X[:, 24*2:24*2+2]
        self.widths = np.linalg.norm(l_hip_pos - r_hip_pos, axis=1)

        # Distance from foot to hip
        self.l_leg_arr = np.linalg.norm(l_hip_pos - l_foot_pos, axis=1)
        self.r_leg_arr = np.linalg.norm(r_hip_pos - r_foot_pos, axis=1)

        # Hand - nose difference in y
        self.l_hny = X[:,0*2+1] - X[:,19*2+1]
        self.r_hny = X[:,0*2+1] - X[:,20*2+1]


        return self
    
    @staticmethod
    def update_skeleton_features(feat_arr):
        feat_arr = list(feat_arr)

        featkeep = [13,14,15,16,17,18,19,20,21,22,25,26,27,28]
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
        X_transformed = X.copy()

        normalized_r_arm = self.r_arm_arr / self.heights
        normalized_l_arm = self.l_arm_arr / self.heights

        normalized_r_leg=self.r_leg_arr /self.heights
        normalized_l_leg=self.l_leg_arr / self.heights

        normalized_r_hand_std = X_transformed[:,20*4+1]/self.widths
        normalized_l_hand_std = X_transformed[:,19*4+1]/self.widths

        X_transformed["Skeleton_Features"] = X_transformed["Skeleton_Features"].apply(self.update_skeleton_features)

        for i, feats in enumerate(X_transformed["Skeleton_Features"]):
            feats.extend([normalized_l_arm[i], normalized_r_arm[i],normalized_l_leg[i],normalized_r_leg[i], normalized_l_hand_std[i], normalized_r_hand_std[i]])


        
        return X_transformed


# %%
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=RANDOM_STATE, stratify=Y)


pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('my_pre_processing', PreProcessing()),
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

grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
# %%
