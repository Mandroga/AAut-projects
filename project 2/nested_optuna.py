# nested_optuna_cv.py

#%%
%run data_imports.py
#%%
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline
import optuna
from keras import layers, callbacks, Model
from keras.optimizers import Adam, RMSprop, Nadam

# ---- Load your data ----
# Make sure X_np (features), Y (labels), groups_all (patient IDs) exist
# Example:
X_np = np.stack(df["Skeleton_Features"].values)
Y = np.array(df["target"].values)
groups_all = df["Patient_Id"].to_numpy()

# ---- Helper functions/classes from your previous code ----
# reorder_fn, normalize_fn, PreProcessingClass, FeatureTransform3Class, build_mlp_from_params

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
#%%

# Shortcut references for the helper functions you defined earlier:
reorder_fn = globals().get("reorder_means_then_stds_to_interleaved")
normalize_fn = globals().get("normalize_center_and_scale")
PreProcessingClass = globals().get("PreProcessing")
FeatureTransform3Class = globals().get("FeatureTransform_np")
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

# ------------------------------------------------------------------------------





#%%
# ---- Optuna model builder ----
def build_mlp_from_params(params, input_dim, num_classes):
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for i in range(params["n_layers"]):
        units = params[f"units_l{i}"]
        x = layers.Dense(units, activation=params["activation"])(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(params["dropout"])(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    
    model = Model(inp, out)
    
    lr = params.get("lr", 1e-3)
    opt_name = params.get("optimizer", "adam")
    if opt_name == "adam":
        opt = Adam(learning_rate=lr)
    elif opt_name == "rmsprop":
        opt = RMSprop(learning_rate=lr)
    elif opt_name == "nadam":
        opt = Nadam(learning_rate=lr)
    else:
        opt = Adam(learning_rate=lr)
        
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ---- Outer / inner CV parameters ----
outer_split = 14  # number of outer folds
inner_split = 13  # number of inner folds

outer_sgkf = StratifiedGroupKFold(n_splits=outer_split, shuffle=True, random_state=42)

outer_results = []
per_fold_reports = {}
best_params_per_fold = {}

fold_id = 0
for train_val_idx, test_idx in outer_sgkf.split(X_np, Y, groups=groups_all):
    fold_id += 1
    print(f"\n[Outer fold {fold_id}]")
    X_trv, X_te = X_np[train_val_idx], X_np[test_idx]
    y_trv, y_te = Y[train_val_idx], Y[test_idx]
    groups_trv = groups_all[train_val_idx]
    
    # --- inner CV Optuna search ---
    def objective(trial):
        params = {}
        params["n_layers"] = trial.suggest_int("n_layers", 6, 7)
        params["dropout"] = trial.suggest_float("dropout", 0.05, 0.5)
        params["activation"] = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])
        params["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "nadam"])
        params["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        params["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32])
        params["epochs"] = trial.suggest_int("epochs", 20, 50)
        
        for i in range(params["n_layers"]):
            params[f"units_l{i}"] = trial.suggest_int(f"units_l{i}", 64, 256, step=64)
        
        # Inner CV
        inner_sgkf = StratifiedGroupKFold(n_splits=inner_split, shuffle=True, random_state=42)
        f1_scores = []
        for inner_train_idx, val_idx in inner_sgkf.split(X_trv, y_trv, groups_trv):
            X_train_inner, X_val_inner = X_trv[inner_train_idx], X_trv[val_idx]
            y_train_inner, y_val_inner = y_trv[inner_train_idx], y_trv[val_idx]
            
            # Preprocessing
            X_train_reordered = reorder_fn(X_train_inner)
            X_val_reordered = reorder_fn(X_val_inner)
            X_train_norm, _, _ = normalize_fn(X_train_reordered)
            X_val_norm, _, _ = normalize_fn(X_val_reordered)
            
            preproc = PreProcessingClass()
            X_train_extra = preproc.transform(X_train_inner)
            X_val_extra = preproc.transform(X_val_inner)
            
            ft = FeatureTransform3Class()
            X_train_extra = ft.fit_transform(X_train_extra)
            X_val_extra = ft.transform(X_val_extra)
            
            X_train_combined = np.hstack([X_train_norm, X_train_extra])
            X_val_combined = np.hstack([X_val_norm, X_val_extra])
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_combined)
            X_val_scaled = scaler.transform(X_val_combined)
            
            num_classes = len(np.unique(y_train_inner))
            model = build_mlp_from_params(params, X_train_scaled.shape[1], num_classes)
            
            cb = [callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)]
            
            model.fit(X_train_scaled, y_train_inner,
                      validation_data=(X_val_scaled, y_val_inner),
                      epochs=params["epochs"],
                      batch_size=params["batch_size"],
                      verbose=0,
                      callbacks=cb)
            
            y_pred = model.predict(X_val_scaled).argmax(axis=1)
            f1_scores.append(f1_score(y_val_inner, y_pred, average="macro"))
        
        return np.mean(f1_scores)
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=10)  
    
    best_params = study.best_trial.params
    best_params_per_fold[f'fold_{fold_id}'] = best_params
    
    # Retrain best model on full train_val set
    X_trv_reordered = reorder_fn(X_trv)
    X_trv_norm, _, _ = normalize_fn(X_trv_reordered)
    preproc = PreProcessingClass()
    X_trv_extra = preproc.transform(X_trv)
    ft = FeatureTransform3Class()
    X_trv_extra = ft.fit_transform(X_trv_extra)
    X_trv_combined = np.hstack([X_trv_norm, X_trv_extra])
    
    scaler = StandardScaler()
    X_trv_scaled = scaler.fit_transform(X_trv_combined)
    
    num_classes = len(np.unique(y_trv))
    model = build_mlp_from_params(best_params, X_trv_scaled.shape[1], num_classes)
    
    cb = [callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)]
    X_tr, X_val, y_tr, y_val = train_test_split(X_trv_scaled, y_trv, test_size=0.1, stratify=y_trv, random_state=42)
    
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=best_params["epochs"],
              batch_size=best_params["batch_size"],
              verbose=0,
              callbacks=cb)
    
    # Evaluate on outer test set
    X_te_reordered = reorder_fn(X_te)
    X_te_norm, _, _ = normalize_fn(X_te_reordered)
    preproc = PreProcessingClass()
    X_te_extra = preproc.transform(X_te)
    ft = FeatureTransform3Class()
    X_te_extra = ft.transform(X_te_extra)
    X_te_combined = np.hstack([X_te_norm, X_te_extra])
    X_te_scaled = scaler.transform(X_te_combined)
    
    y_pred = model.predict(X_te_scaled).argmax(axis=1)
    f1_macro = f1_score(y_te, y_pred, average="macro")
    outer_results.append(f1_macro)
    
    per_fold_reports[f'fold_{fold_id}'] = classification_report(y_te, y_pred, digits=3)
    print(f"Fold {fold_id} F1-macro: {f1_macro:.4f}")

# ---- Summary ----
outer_results = np.array(outer_results)
print("\n[Nested SGKF] F1-macro per fold:", np.round(outer_results, 4))
print(f"[Nested SGKF] Mean ± Std: {outer_results.mean():.4f} ± {outer_results.std():.4f}")

# %%
