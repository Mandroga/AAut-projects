
#%%

from tkinter import Y
from turtle import width

from NN import X


%run data_imports.py

# %% 


y = np.array(df['target'].to_list())

hands= [19,20]
X_nhposy = np.column_stack([
    X[:, 0*2+1] - X[:, 19*2+1],
    X[:, 0*2+1] - X[:, 20*2+1]
])

#%%
print(X_nhposy.shape)

#%%
featkeep = [13,14,17,18,25,26,27,28]

Xi=np.array([X[:,i*2] for i in range(len(X)) if i in featkeep]+[X[:,i*2+1] for i in range(len(X)) if i in featkeep]+[ X[:,i*4] for i in range(len(X)) if i in featkeep]+[X[:,i*4+1] for i in range(len(X)) if i in featkeep])   

Xi = Xi.T  # shape becomes (700, 24)


my_X = np.hstack([Xi, X_nhposy])


my_X_df = pd.DataFrame({
    'Patient_Id': df['Patient_Id'],
    'Features': list(my_X)  

print(my_X_df)
#%%



X= np.array(df['Skeleton_Features'].to_list())
Y=np.load("Ytrain1.npy")
#%%    

from scipy.spatial.distance import euclidean

class PreProcessing(BaseEstimator, TransformerMixin):
    """
    Custom preprocessing transformer for scikit-learn pipelines.
    Implements fit/transform methods so it can be used in a Pipeline.
    """

    def __init__(self, param1=None, param2=None):
        """
        Initialize any hyperparameters here.
        They will be stored as attributes.
        Example: param1 = scaling factor, param2 = threshold, etc.
        """
        self.l_arm_dic = {}
        self.r_arm_dic= {}
        self.heights ={}
        self.widths={}

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

        self.l_arm_dic = {}
        self.r_arm_dic = {}
            for index, feat_arr in enumerate(X):
            
                #Distance from shoulder to hand
                l_shoulder_pos, l_hand_pos= (feat_arr[11*2],feat_arr[11*2+1]),(feat_arr[19*2],feat_arr[19*2+1])
                r_shoulder_pos, r_hand_pos = (feat_arr[12*2],feat_arr[12*2+1]),(feat_arr[20*2],feat_arr[20*2+1])
                
                l_arm = euclidean(l_shoulder_pos,l_hand_pos)
                r_arm = euclidean(r_shoulder_pos, r_hand_pos)
                self.l_arm_dic[index] = l_arm
                self.r_arm_dic[index]= r_arm

                #Height
                l_eye_pos, l_foot_pos= (feat_arr[3*2],feat_arr[3*2+1]),(feat_arr[29*2],feat_arr[29*2+1]) 
                r_eye_pos, r_foot_pos = (feat_arr[6*2],feat_arr[6*2+1]),(feat_arr[30*2],feat_arr[30*2+1])

                l_height = euclidean(l_eye_pos,l_foot_pos)
                r_height = euclidean(r_eye_pos, r_foot_pos)
                height = max(l_height, r_height)
                self.heights[index]= height

                #Width
                l_hip_pos, r_hip_pos= (feat_arr[23*2],feat_arr[23*2+1]),(feat_arr[24*2],feat_arr[24*2+1]) 
                width=euclidean(l_hip_pos, r_hip_pos)
                self.widths[index]=width



        return self
    
    @staticmethod
    def update_skeleton_features(feat_arr):
        feat_arr = list(feat_arr)
        #hand nose y difference
        feat_arr_add = [
            feat_arr[0*2+1] - feat_arr[19*2+1],
            feat_arr[0*2+1] - feat_arr[20*2+1]
        ]

        featkeep = [13,14,17,18,25,26,27,28]
        feat_arr = [feat_arr[i*2] for i in featkeep]+[feat_arr[i*2+1]for i in featkeep]+[feat_arr[i*4] for i in featkeep]+[feat_arr[i*4+1]for i in featkeep]+feat_arr_add


        return feat_arr

    @staticmethod
    def extra_features(feat_arr):
        feat_arr = list(feat_arr)
        #hand nose y difference
        feat_arr += [
            feat_arr[0*2+1] - feat_arr[19*2+1],
            feat_arr[0*2+1] - feat_arr[20*2+1]
        ]
    
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
        X_transformed["Skeleton_Features"] = X_transformed["Skeleton_Features"].apply(self.update_skeleton_features)
        X_transformed["Skeleton_Features"] = X_transformed["Skeleton_Features"].apply(self.extra_features)

        for row in X_transformed.itertuples(index=True):
            patient = row.Patient_Id
            feat=row.Skeleton_Features
            index=row.index

            normalized_r_hand=self.r_arm_dic[index]/self.maximum_arms_length[patient]
            normalized_l_hand=self.l_arm_dic[index]/self.maximum_arms_length[patient]
            feat+=[normalized_r_hand, normalized_l_hand]

        
        return X_transformed


# %%
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('my_pre_processing', PreProcessing())
    ('mlp', MLPClassifier(max_iter=500, random_state=RANDOM_STATE, early_stopping=True,validation_fraction=0.1))
])

param_grid = {
    'mlp__hidden_layer_sizes': [(50,), (100,), (100,50)],
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
