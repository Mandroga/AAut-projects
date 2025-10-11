#%%

%run data_imports.py

X_raw = np.stack(df["Skeleton_Features"].values) # shape (n_samples, n_features)  
X = X_raw.copy()
y = np.array(df["target"].values)
df['weights'] = w
groups = df['Patient_Id'].values             # patient IDs
#%%

import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, callbacks, optimizers
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import kerastuner as kt


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
        X = X.copy()
        for i in range(33):
            X[:, i*2+1] = -X[:, i*2+1]
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

#%%

# -------------------------
# 1) Define a tunable MLP builder
# -------------------------
def build_mlp_tunable(hp, input_dim, num_classes):
    """
    Build a Keras MLP with tunable hyperparameters:
    - number of layers (1-4)
    - units per layer
    - activation
    - dropout
    """
    inp = layers.Input(shape=(input_dim,))

    x = inp
    n_layers = hp.Int("n_layers", 3, 7, default=3)
    for i in range(n_layers):
        units = hp.Choice(f"units_{i}", [64, 128, 256, 512], default=128)
        x = layers.Dense(units, activation=None)(x)
        if hp.Boolean(f"batchnorm_{i}", True):
            x = layers.BatchNormalization()(x)
        act = hp.Choice(f"activation_{i}", ["relu", "sigmoid", "swish", "tanh"], default="tanh")
        x = layers.Activation(act)(x)
        dropout = hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.05, default=0.3)
        if dropout > 0.0:
            x = layers.Dropout(dropout)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out)

    # Optimizer tuning
    opt_name = hp.Choice("optimizer", ["adam", "rmsprop", "sgd"], default="adam")
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log", default=1e-3)
    if opt_name == "adam":
        optimizer = optimizers.Adam(learning_rate=lr)
    elif opt_name == "rmsprop":
        optimizer = optimizers.RMSprop(learning_rate=lr)
    else:
        optimizer = optimizers.SGD(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -------------------------
# 2) Callback to compute macro-F1
# -------------------------
from sklearn.metrics import f1_score

class F1Callback(callbacks.Callback):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        self.best_f1 = 0.0

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        f1 = f1_score(self.y_val, y_pred, average='macro')
        logs['val_f1'] = f1
        if f1 > self.best_f1:
            self.best_f1 = f1
        print(f" — val_f1: {f1:.4f}")

# -------------------------
# 3) Training per fold with tuner
# -------------------------
def train_with_tuner(X_train, y_train, X_val, y_val, num_classes,
                     max_trials=10, epochs=50, batch_size=16, fold_id=0):
    """
    Build tuner after seeing X_train shape. Returns best_model and best val-F1.
    """
    input_dim = X_train.shape[1]   # <-- infer AFTER preprocessing!!

    # closure that uses input_dim known from X_train
    def tuner_model(hp):
        return build_mlp_tunable(hp, input_dim=input_dim, num_classes=num_classes)

    tuner = kt.RandomSearch(
        tuner_model,
        objective=kt.Objective("val_f1", direction="max"),
        max_trials=max_trials,
        executions_per_trial=1,
        directory=f"tuner_dir_fold{fold_id}",           # per-fold directory
        project_name=f"mlp_fold_{fold_id}",
        # make the oracle more tolerant of failing trials:
        distribution_strategy=None
    )

    # increase allowable consecutive failures (if your keras-tuner supports setting it)
    try:
        tuner.oracle.max_consecutive_failed_trials = 10
    except Exception:
        # older/newer kt might not expose this — ignore if not available
        pass

    f1_cb = F1Callback(X_val, y_val)
    es = callbacks.EarlyStopping(monitor="val_f1", mode='max', patience=10, restore_best_weights=True)

    # Run search (tuner will build models using the correct input_dim)
    tuner.search(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=epochs,
                 batch_size=batch_size,
                 callbacks=[f1_cb, es],
                 verbose=1)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.hypermodel.build(best_hp)

    # Fit final model (with same callbacks so val_f1 gets monitored and best weights restored)
    best_model.fit(X_train, y_train,
                   validation_data=(X_val, y_val),
                   epochs=epochs,
                   batch_size=batch_size,
                   callbacks=[f1_cb, es],
                   verbose=1)

    y_pred = np.argmax(best_model.predict(X_val), axis=1)
    f1 = f1_score(y_val, y_pred, average='macro')
    print("Fold F1:", f1)
    return best_model, f1

# -------------------------
# 4) Main cross-validation loop
# -------------------------
# updated cross_val_tuned_mlp: preprocess BEFORE calling train_with_tuner
def cross_val_tuned_mlp(X, y, groups, n_splits=3, max_trials=5, epochs=50, batch_size=16):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results_list = []
    val_patients_all = []

    num_classes = len(np.unique(y))

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"Fold {fold}")
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        train_patients = groups[train_idx]
        val_patients = groups[val_idx]
        print("Train patient IDs:", np.unique(train_patients))
        print("Validation patient IDs:", np.unique(val_patients))

        # ---- YOUR preprocessing (exactly like before) ----
        preproc = PreProcessing()
        featrans = FeatureTransform_np()

        X_train_proc = preproc.transform(X_train_raw)
        X_val_proc   = preproc.transform(X_val_raw)

        X_train_proc = featrans.fit_transform(X_train_proc)
        X_val_proc   = featrans.transform(X_val_proc)

        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train_proc)
        X_val_proc = scaler.transform(X_val_proc)

        # --- now call tuner (train_with_tuner infers input_dim from X_train_proc) ---
        best_model, f1 = train_with_tuner(
            X_train_proc, y_train, X_val_proc, y_val,
            num_classes=num_classes,
            max_trials=max_trials, epochs=epochs, batch_size=batch_size,
            fold_id=fold
        )

        results_list.append(f1)
        val_patients_all += list(np.unique(val_patients)),
    print("Best models", best_model.summary())
    print("Average F1 over folds:", np.mean(results_list))
    print("Max F1 over folds:", max(results_list))
    print("Validation patient IDs over all folds:", val_patients_all)
    return results_list

# -------------------------
# 5) Example usage
# -------------------------
if __name__ == "__main__":
    # load X, y, groups from your data_imports.py
    from data_imports import df, w
    X_raw = np.stack(df["Skeleton_Features"].values)
    X = X_raw.copy()
    y = np.array(df["target"].values)
    groups = df['Patient_Id'].values

    f1_scores = cross_val_tuned_mlp(X, y, groups, 
                                    n_splits=3,
                                    max_trials=5,  # reduce for quick testing
                                    epochs=30,
                                    batch_size=16)
    print("F1 scores per fold:", f1_scores)

# %%
