#%%

%run data_imports.py

X_raw = np.stack(df["Skeleton_Features"].values) # shape (n_samples, n_features)  
X = X_raw.copy()
y = np.array(df["target"].values)
df['weights'] = w
groups = df['Patient_Id'].values             # patient IDs
#%%

n_splits = 3
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
#%%
import kerastuner as kt  
from tensorflow.keras import layers, callbacks
from sklearn.metrics import f1_score
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras import layers, models, callbacks,regularizers, optimizers
from sklearn.model_selection import StratifiedGroupKFold

from data_imports import FeatureTransform, FeatureTransform_np
import kerastuner as kt 
#from tensorflow_addons.metrics import FBetaScore

# from keras.optimizers import Adam



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


#%%
# -----------------------
# 4) standardize (fit on train only)
# -----------------------
def fit_standardizer(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def apply_standardizer(X, scaler):
    return scaler.transform(X)

#%%
# -----------------------
# 5) quick models: MLP and RandomForest baselines
# -----------------------
def build_mlp(input_dim, num_classes, dropout=0.3):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='sigmoid')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='sigmoid')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model = models.Model(inp, out)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate(X_train_raw, X_test_raw, y_train, y_test,
                       w_train=None, w_test=None,
                       epochs=100, batch_size=16):
    """
    X_train_raw, X_test_raw: (n_samples,132) raw layout
    y_train, y_test: array-like
    w_train, w_test: arrays of shape (n_samples,) or None
    Returns trained models and scaler
    """
    # 0) If weights are None, make them all ones
    if w_train is None:
        w_train = np.ones(len(y_train), dtype=np.float32)
    if w_test is None:
        w_test = np.ones(len(y_test), dtype=np.float32)

    # 1) reorder to interleaved
    # Original skeleton features (means-then-std layout)
    X_train_orig = reorder_means_then_stds_to_interleaved(X_train_raw)
    X_test_orig  = reorder_means_then_stds_to_interleaved(X_test_raw)

    # 2) normalize (safe defaults: median root centering + torso fallback scaling)
    X_train_norm, _, _ = normalize_center_and_scale(X_train_orig,
                                                    center_mode="root",
                                                    root_index=None,
                                                    scale_mode="torso",
                                                    torso_pair=None)
    X_test_norm, _, _ = normalize_center_and_scale(X_test_orig,
                                                center_mode="root",
                                                root_index=None,
                                                scale_mode="torso",
                                                torso_pair=None)

    scaler_orig = StandardScaler()
    
    X_train_skel = scaler_orig.fit_transform(X_train_norm)
    X_test_skel  = scaler_orig.transform(X_test_norm)

    preproc = PreProcessing()

    X_train_extra = preproc.transform(X_train_raw)
    X_test_extra  = preproc.transform(X_test_raw)

    featrans = FeatureTransform_np()
    X_train_extra = featrans.fit_transform(X_train_extra)
    X_test_extra  = featrans.transform(X_test_extra)

    
    X_train_combined = np.hstack([X_train_skel, X_train_extra])
    X_test_combined  = np.hstack([X_test_skel,  X_test_extra])


    #X_train_norm, scales, centers = normalize_center_and_scale(X_train,
#                                                               center_mode="root",
#                                                               root_index=None,
#                                                               scale_mode="torso",
#                                                               torso_pair=None)
    #X_test_norm, _, _ = normalize_center_and_scale(X_test,
#                                                   center_mode="root",
#                                                   root_index=None,
#                                                   scale_mode="torso",
#                                                   torso_pair=None)

    # 3) standardize on train
    #scaler = fit_standardizer(X_train_norm)
    #X_train_s = apply_standardizer(X_train_norm, scaler)
    #X_test_s  = apply_standardizer(X_test_norm, scaler)

    # 4a) train RandomForest baseline
    rf = RandomForestClassifier(n_estimators=200, random_state=0)
    rf.fit(X_train_combined, y_train)
    rf_pred = rf.predict(X_test_combined)
    print("RandomForest report:")
    print(classification_report(y_test, rf_pred))

    # 4b) train MLP
    num_classes = len(np.unique(y_train))  # number of exercise classes
    mlp = build_mlp(X_train_combined.shape[1], num_classes)
    cb = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5)
    ]
    mlp.fit(X_train_combined, y_train, validation_data=(X_test_combined, y_test),
            epochs=epochs, sample_weight=w_train, batch_size=batch_size, callbacks=cb, verbose=1)
    mlp_pred = mlp.predict(X_test_combined).argmax(axis=1)
    print("MLP report:")
    print(classification_report(y_test, mlp_pred))

    # confusion matrices
    print("RF confusion matrix:\n", confusion_matrix(y_test, rf_pred))
    print("MLP confusion matrix:\n", confusion_matrix(y_test, mlp_pred))
    
    # F1 score
    f1 = f1_score(y_test, mlp_pred, average='macro')
    print("F1 score MLP:", f1)

    return f1 #,{'mlp_model': mlp, 'rf_model': rf, 'scaler': scaler_orig}


# %%
n_repeats= 2
results_list = []
val_patients_all = []
sgkf = StratifiedGroupKFold(n_splits=14, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    X_train_raw, X_val_raw = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    w_train = np.array(df['weights'].values[train_idx])
    w_val   = np.array(df['weights'].values[val_idx])
    
    train_patients = groups[train_idx]
    val_patients = groups[val_idx]
    print("Fold {fold}")
    print("Train patient IDs:", np.unique(train_patients))
    print("Validation patient IDs:", np.unique(val_patients))

    results = train_and_evaluate(X_train_raw, X_val_raw, y_train, y_val,
                                    #w_train=w_train,
                                    epochs=50, batch_size=16)
    results_list.append(results)
    val_patients_all += list(np.unique(val_patients)),
print("Average F1 over folds:", np.mean(results_list, axis=0))
print("Max F1 over folds", max(results_list))
print("Validation patient IDs over all folds:", val_patients_all)
# %%
"""
            normalized_l_arm,
            normalized_r_arm,
            normalized_l_leg,
            normalized_r_leg,
            normalized_l_hand_std,
            normalized_r_hand_std,
"""
# %%
