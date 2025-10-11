# optuna_tuning_save_best.py
#%%


%run data_imports.py
X_raw = np.stack(df["Skeleton_Features"].values) # shape (n_samples, n_features)  
X = X_raw.copy()
y = np.array(df["target"].values)
df['weights'] = w
groups = df['Patient_Id'].values             # patient IDs
#%%"



import os
import pickle
import joblib
import optuna
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import keras
from keras import layers, callbacks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---- Adjust these imports if your functions are in a different module ----
# The code you posted defines: reorder_means_then_stds_to_interleaved,
# normalize_center_and_scale, PreProcessing, FeatureTransform3, and variables X, y, df, groups.
# If those live in a module file, import them. Example:
# from data_processing_file import reorder_means_then_stds_to_interleaved, normalize_center_and_scale, PreProcessing
# from imports import FeatureTransform3
# If they are already defined in the current environment (e.g. in the notebook), no import is needed.

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
EPS = 1e-9

# -----------------------
# 1) reorder means+stds -> interleaved per keypoint (x,y,x_std,y_std)
# -----------------------
def reorder_means_then_stds_to_interleaved(X_raw):
    """
    X_raw: (N,132) with layout [x1,y1,...,x33,y33, x1_std,y1_std,...,x33_std,y33_std]
    Returns: X_int (N,132) with layout [x1,y1,x1_std,y1_std, x2,y2,x2_std,y2_std, ...]
    """
    X_raw = np.asarray(X_raw)
    if X_raw.shape[1] != 132:
        raise ValueError("Expected 132 features per row.")
    N = X_raw.shape[0]
    # split means and stds
    means = X_raw[:, :66]   # order: x1,y1,...,x33,y33
    stds  = X_raw[:, 66:]   # order: x1_std,y1_std,...
    # reshape to (N,33,2)
    means_kp = means.reshape(N, 33, 2)
    stds_kp  = stds.reshape(N, 33, 2)
    # interleave into (N,33,4) -> flatten to (N,132)
    interleaved = np.concatenate([means_kp, stds_kp], axis=2)  # (N,33,4) columns: x,y,xstd,ystd
    X_int = interleaved.reshape(N, 33*4)
    return X_int

# -----------------------
# reuse normalization utilities from before (centering + scaling)
# -----------------------
def split_132_to_components(X):
    x_means = X[:, 0::4]   # (N,33)
    y_means = X[:, 1::4]
    x_stds  = X[:, 2::4]
    y_stds  = X[:, 3::4]
    return x_means, y_means, x_stds, y_stds

def merge_components(xm, ym, xs, ys):
    N = xm.shape[0]
    F = np.zeros((N, 33*4), dtype=np.float32)
    for k in range(33):
        F[:, k*4 + 0] = xm[:, k]
        F[:, k*4 + 1] = ym[:, k]
        F[:, k*4 + 2] = xs[:, k]
        F[:, k*4 + 3] = ys[:, k]
    return F

def normalize_center_and_scale(X,
                               center_mode="root",   # "root" or "centroid"
                               root_index=None,      # if you know a root joint index, pass it; otherwise None -> median
                               scale_mode="torso",   # "torso", "bbox", "std", or "none"
                               torso_pair=None       # tuple of indices (i1,i2) if you want explicit torso pair
                              ):
    """
    Returns X_norm (N,132), scales (N,), centers (N,2)
    Uses safe defaults: root median + torso fallback to mean magnitude.
    """
    xm, ym, xs, ys = split_132_to_components(X)
    N = X.shape[0]

    # centers
    if center_mode == "centroid":
        cx = np.nanmean(xm, axis=1)
        cy = np.nanmean(ym, axis=1)
    else:  # root
        if root_index is None:
            cx = np.nanmedian(xm, axis=1)
            cy = np.nanmedian(ym, axis=1)
        else:
            cx = xm[:, root_index]
            cy = ym[:, root_index]

    xm_c = xm - cx[:, None]
    ym_c = ym - cy[:, None]

    # scale
    if scale_mode == "none":
        scales = np.ones(N, dtype=np.float32)
    elif scale_mode == "bbox":
        x_span = np.nanmax(xm_c, axis=1) - np.nanmin(xm_c, axis=1)
        y_span = np.nanmax(ym_c, axis=1) - np.nanmin(ym_c, axis=1)
        scales = np.maximum(x_span, y_span)
    elif scale_mode == "std":
        coords = np.concatenate([xm_c, ym_c], axis=1)
        scales = np.std(coords, axis=1)
    elif scale_mode == "torso":
        if torso_pair is not None:
            i1, i2 = torso_pair
            dx = xm_c[:, i1] - xm_c[:, i2]
            dy = ym_c[:, i1] - ym_c[:, i2]
            scales = np.sqrt(dx*dx + dy*dy)
        else:
            # fallback: mean magnitude of keypoints
            coords = np.stack([xm_c, ym_c], axis=-1)  # (N,33,2)
            mag = np.mean(np.sqrt(np.sum(coords**2, axis=-1)), axis=1)
            scales = mag
    else:
        raise ValueError("unknown scale_mode")

    scales = np.where(np.abs(scales) < EPS, 1.0, scales).astype(np.float32)

    xm_cs = xm_c / scales[:, None]
    ym_cs = ym_c / scales[:, None]
    xs_cs = xs / scales[:, None]
    ys_cs = ys / scales[:, None]

    X_norm = merge_components(xm_cs, ym_cs, xs_cs, ys_cs)
    centers = np.stack([cx, cy], axis=1)
    return X_norm, scales, centers
# If the helper functions are in the same script / notebook they will be used directly.
# ------------------------------------------------------------------------------

# ---- Settings ----
N_TRIALS = 40             # change to a larger value for more search
CV_SPLITS = 3             # keep small to speed up during tuning
RANDOM_STATE = 42

# Output filenames
OUT_DIR = "optuna_results"
os.makedirs(OUT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_mlp.h5")
SCALER_PATH = os.path.join(OUT_DIR, "scaler.pkl")
STUDY_PATH = os.path.join(OUT_DIR, "optuna_study.pkl")

# ---- Make sure X, y, df, groups exist in the environment ----
# The script expects the variables to be available (from your earlier code).
# If not, load them here (e.g., import data_imports or run your data-loading cell).
if "X" not in globals() or "y" not in globals() or "df" not in globals() or "groups" not in globals():
    raise RuntimeError("This script expects X, y, df and groups to exist in the environment. "
                       "Run your data-loading cell (or import them) before running this file.")

# Shortcut references for the helper functions you defined earlier:
reorder_fn = globals().get("reorder_means_then_stds_to_interleaved")
normalize_fn = globals().get("normalize_center_and_scale")
PreProcessingClass = globals().get("PreProcessing")
FeatureTransform3Class = globals().get("FeatureTransform_np")

if reorder_fn is None or normalize_fn is None or PreProcessingClass is None or FeatureTransform3Class is None:
    raise RuntimeError("Required preprocessing helpers not found in globals(): "
                       "reorder_means_then_stds_to_interleaved, normalize_center_and_scale, "
                       "PreProcessing, FeatureTransform3")

# ---- Model builder for Optuna ----
def build_mlp_from_params(params, input_dim, num_classes):
    """
    Build a compiled Keras model from hyperparameter dict `params`.
    """
    inp = layers.Input(shape=(input_dim,))
    x = inp
    n_layers = params["n_layers"]
    activation = params["activation"]
    dropout = params["dropout"]
    for i in range(n_layers):
        units = params[f"units_l{i}"]
        x = layers.Dense(units, activation=activation)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out)

    # optimizer selection
    lr = params.get("lr", 1e-3)
    opt_name = params.get("optimizer", "adam")
    if opt_name == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    elif opt_name == "nadam":
        opt = keras.optimizers.Nadam(learning_rate=lr)
    else:
        opt = keras.optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ---- Optuna objective ----
def objective(trial):
    # propose hyperparameters
    params = {}
    params["n_layers"] = trial.suggest_int("n_layers", 2, 8)
    params["dropout"] = trial.suggest_float("dropout", 0.05, 0.6)
    params["activation"] = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh", "swish"])
    params["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "nadam"])
    params["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32])
    params["epochs"] = trial.suggest_int("epochs", 20, 80)

    # units per layer
    for i in range(params["n_layers"]):
        params[f"units_l{i}"] = trial.suggest_int(f"units_l{i}", 64, 512, step=64)

    # cross-validation over groups
    sgkf = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    f1_scores = []

    fold_idx = 0
    for train_idx, val_idx in sgkf.split(X, y, groups):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 1) Reorder + normalize (use your helpers)
        X_train_reordered = reorder_fn(X_train_raw)
        X_val_reordered   = reorder_fn(X_val_raw)

        X_train_norm, _, _ = normalize_fn(X_train_reordered,
                                          center_mode="root",
                                          root_index=None,
                                          scale_mode="torso",
                                          torso_pair=None)
        X_val_norm, _, _ = normalize_fn(X_val_reordered,
                                        center_mode="root",
                                        root_index=None,
                                        scale_mode="torso",
                                        torso_pair=None)

        # 2) engineered features
        preproc = PreProcessingClass()
        X_train_extra = preproc.transform(X_train_raw)
        X_val_extra = preproc.transform(X_val_raw)

        ft = FeatureTransform3Class()
        X_train_extra = ft.fit_transform(X_train_extra)
        X_val_extra = ft.transform(X_val_extra)

        # 3) combine
        X_train_combined = np.hstack([X_train_norm, X_train_extra])
        X_val_combined   = np.hstack([X_val_norm, X_val_extra])

        # 4) standardize (fit on train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_combined)
        X_val_scaled = scaler.transform(X_val_combined)

        # 5) baseline RF (optional, quick check) - disabled for speed, enable if you like
        # rf = RandomForestClassifier(n_estimators=200, random_state=0)
        # rf.fit(X_train_scaled, y_train)
        # rf_pred = rf.predict(X_val_scaled)
        # print("RF F1:", f1_score(y_val, rf_pred, average='macro'))

        # 6) build model and train
        num_classes = len(np.unique(y_train))
        model = build_mlp_from_params(params, X_train_scaled.shape[1], num_classes)

        cb = [
            callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
        ]

        history = model.fit(
            X_train_scaled,
            y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0,
            callbacks=cb
        )

        y_pred = model.predict(X_val_scaled).argmax(axis=1)
        f1 = f1_score(y_val, y_pred, average="macro")
        f1_scores.append(f1)

        # prune if trial is poor
        trial.report(np.mean(f1_scores), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
        fold_idx += 1

    mean_f1 = float(np.mean(f1_scores))
    print(f"[Trial {trial.number}] mean_f1 = {mean_f1:.4f}, params = {params}")
    return mean_f1

# ---- Run study ----
def run_optuna_search(n_trials=N_TRIALS):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    
    study.optimize(objective, n_trials=n_trials)
    # save study
    with open(STUDY_PATH, "wb") as f:
        pickle.dump(study, f)
    print("Study saved to:", STUDY_PATH)
    return study

# ---- Retrain best model on entire dataset and save model + scaler ----
def retrain_and_save_best(study):
    best_params = study.best_trial.params
    # adapt keys: ensure units fields exist as ints
    # build param dict expected by build_mlp_from_params
    params = {}
    params["n_layers"] = best_params["n_layers"]
    params["dropout"] = best_params["dropout"]
    params["activation"] = best_params["activation"]
    params["optimizer"] = best_params["optimizer"]
    params["lr"] = best_params["lr"]
    params["batch_size"] = best_params["batch_size"]
    params["epochs"] = int(best_params["epochs"])
    for i in range(params["n_layers"]):
        params[f"units_l{i}"] = int(best_params[f"units_l{i}"])

    # Preprocess entire dataset
    X_reordered = reorder_fn(X)
    X_norm, _, _ = normalize_fn(X_reordered,
                                center_mode="root",
                                root_index=None,
                                scale_mode="torso",
                                torso_pair=None)
    preproc = PreProcessingClass()
    X_extra = preproc.transform(X)
    ft = FeatureTransform3Class()
    X_extra = ft.fit_transform(X_extra)
    X_combined = np.hstack([X_norm, X_extra])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    num_classes = len(np.unique(y))
    model = build_mlp_from_params(params, X_scaled.shape[1], num_classes)

    cb = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
    ]

    # split a small validation slice to let early stopping work during final training
    # use stratified split
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.12, stratify=y, random_state=RANDOM_STATE)

    model.fit(X_tr, y_tr,
              validation_data=(X_val, y_val),
              epochs=params["epochs"],
              batch_size=params["batch_size"],
              callbacks=cb,
              verbose=1)

    # Save model and scaler
    model.save(BEST_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Saved Keras model to:", BEST_MODEL_PATH)
    print("Saved scaler to:", SCALER_PATH)

# ---- Main entrypoint ----
if __name__ == "__main__":
    print("Starting Optuna search (this will take a while)...")
    study = run_optuna_search(n_trials=N_TRIALS)
    print("Best trial:", study.best_trial.number, "F1:", study.best_trial.value)
    print("Best params:", study.best_trial.params)

    print("Retraining best model on entire dataset and saving artifacts...")
    retrain_and_save_best(study)
    print("All done. Artifacts are in:", os.path.abspath(OUT_DIR))

# %%
if __name__ == "__main__":
    print("Starting Optuna search (this will take a while)...")

    # ---------- Diagnostic: run one CV split and print shapes ----------
    from sklearn.model_selection import StratifiedGroupKFold
    print("Running one diagnostic CV split to inspect shapes...")

    sgkf = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(iter(sgkf.split(X, y, groups)))
    X_train_raw, X_val_raw = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # reorder + normalize
    X_train_reordered = reorder_fn(X_train_raw)
    X_val_reordered   = reorder_fn(X_val_raw)
    print("reordered shapes:", X_train_reordered.shape, X_val_reordered.shape)

    X_train_norm, _, _ = normalize_fn(X_train_reordered,
                                      center_mode="root",
                                      root_index=None,
                                      scale_mode="torso",
                                      torso_pair=None)
    X_val_norm, _, _ = normalize_fn(X_val_reordered,
                                    center_mode="root",
                                    root_index=None,
                                    scale_mode="torso",
                                    torso_pair=None)
    print("normalized shapes:", X_train_norm.shape, X_val_norm.shape)

    # engineered features
    preproc = PreProcessingClass()
    X_train_extra = preproc.transform(X_train_raw)
    X_val_extra = preproc.transform(X_val_raw)
    print("preproc outputs types/shapes:", type(X_train_extra), getattr(X_train_extra, "shape", None), type(X_val_extra), getattr(X_val_extra, "shape", None))

    # FeatureTransform3
    ft = FeatureTransform3Class()
    X_train_extra_ft = ft.fit_transform(X_train_extra)
    X_val_extra_ft = ft.transform(X_val_extra)
    print("FeatureTransform3 outputs types/shapes (train/val):",
          type(X_train_extra_ft), getattr(X_train_extra_ft, "shape", None),
          type(X_val_extra_ft), getattr(X_val_extra_ft, "shape", None))

    # Force np.asarray and check shapes again
    X_train_extra_ft = np.asarray(X_train_extra_ft)
    X_val_extra_ft   = np.asarray(X_val_extra_ft)
    print("After np.asarray shapes:", X_train_extra_ft.shape, X_val_extra_ft.shape)

    # Combined shapes
    try:
        X_train_combined = np.hstack([X_train_norm, X_train_extra_ft])
        X_val_combined   = np.hstack([X_val_norm,   X_val_extra_ft])
        print("Combined shapes:", X_train_combined.shape, X_val_combined.shape)
    except Exception as e:
        print("Error when hstacking:", e)
        print("X_train_norm.shape, X_train_extra_ft.shape:", X_train_norm.shape, X_train_extra_ft.shape)
        raise

    print("Diagnostic complete — if shapes differ between train and val, fix FeatureTransform3 to produce consistent columns.")
    # ---------- end diagnostic ----------

    # Now run Optuna after diagnostic
    study = run_optuna_search(n_trials=N_TRIALS)
    print("Best trial:", study.best_trial.number, "F1:", study.best_trial.value)
    print("Best params:", study.best_trial.params)

# %%
print("Max F1 over folds", max(study.trials_dataframe()["value"]))
#%%
groups_all = X['Patient_Id'].to_numpy()