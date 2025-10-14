import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import copy
import importlib
import tools
importlib.reload(tools) 
from tools import *
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import trim_mean
from sklearn.base import BaseEstimator, TransformerMixin, clone
import joblib
from sklearn.ensemble import IsolationForest    
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from scipy.spatial.distance import euclidean
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import random
from skopt.space import Space
from sklearn.svm import SVC
RANDOM_STATE = 42
from keras import layers, callbacks
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

# -----------------------
# 4) standardize
# -----------------------
def fit_standardizer(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler

def apply_standardizer(X, scaler):
    return scaler.transform(X)



class FeatureTransform_np(BaseEstimator, TransformerMixin):
    """
    Input X: numpy array with columns in the order:
      [xmean0, ymean0, xmean1, ymean1, ..., xmean32, ymean32,
       xstd0,  ystd0,  xstd1,  ystd1,  ..., xstd32,  ystd32]

    Adds:
      1) Hand-averaged features for ('left','right') across indices in self.keypoints_hands
         -> xmean{key}_hand, ymean{key}_hand, xstd{key}_hand, ystd{key}_hand
      2) Diff features for given pairs (normalized by torso_length):
         -> {component}diff{a}-{b}  for comp in ('xmean','ymean') and (a,b) in self.diff_pairs
            where a/b ∈ {int joint index, 'left_hand', 'right_hand'}
      3) Aggregate stds over left/right sides: left_std, right_std
      4) Torso length = mean of sqrt((11-23)^2) and sqrt((12-24)^2) using xmean/ymean
      5) Passthrough knee stds: xstd25, ystd25, xstd26, ystd26

    Returns: numpy array with columns in a stable, documented order
    """

    def __init__(self):
        # joint indices (0..32)
        self.keypoint_side = {
            'left':  [4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
            'right': [1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31]
        }
        self.keypoints_hands = {'left': [15,17,19,21], 'right': [16,18,20,22]}
        self.diff_pairs = (
            (25,23), (26,24), (27,23), (28,24),
            ('left_hand', 0), ('right_hand', 0),
            ('left_hand', 11), ('right_hand', 12),
            (13,0), (14,0), (13,11), (14,12)
        )
        self.components_hand = ('xmean','ymean','xstd','ystd')
        self.components_diff = ('xmean','ymean')

        self.n_joints_expected = 33  # 0..32
        self.output_features_ = None

    # --------- helpers on numpy layout ----------
    def _split_blocks(self, X):
        """Return (means_block, stds_block), each shape (n_samples, 2*n_joints)."""
        nJ = self.n_joints_expected
        if X.shape[1] < 4*nJ:
            raise ValueError(
                f"Expected at least {4*nJ} columns (got {X.shape[1]}). "
                "Columns must be [xmean0,ymean0,...,xmean32,ymean32, xstd0,ystd0,...,xstd32,ystd32]."
            )
        means_block = X[:, :2*nJ]
        stds_block  = X[:, 2*nJ:2*nJ+2*nJ]  # keep exactly 2*nJ for stds
        return means_block, stds_block

    def _view_comp(self, X, comp):
        """
        Return view of a component with shape (n_samples, n_joints) under the layout:
        [xmean,ymean]*nJ | [xstd,ystd]*nJ
        """
        nJ = self.n_joints_expected
        means_block, stds_block = self._split_blocks(X)

        if comp == 'xmean':
            return means_block[:, 0::2][:, :nJ]
        elif comp == 'ymean':
            return means_block[:, 1::2][:, :nJ]
        elif comp == 'xstd':
            return stds_block[:, 0::2][:, :nJ]
        elif comp == 'ystd':
            return stds_block[:, 1::2][:, :nJ]
        else:
            raise ValueError(f"Unknown component '{comp}'")

    def _hand_feature_names(self):
        return [f"{comp}{key}_hand"
                for key in ('left','right')
                for comp in self.components_hand]

    def _diff_feature_names(self):
        return [f"{comp}diff{a}-{b}"
                for (a,b) in self.diff_pairs
                for comp in self.components_diff]

    def _knee_std_cols(self):
        return [f"{c}{i}" for i in (25, 26) for c in ('xstd','ystd')]

    def fit(self, X, y=None):
        # validate shape
        nJ = self.n_joints_expected
        if X.ndim != 2 or X.shape[1] < 4*nJ:
            raise ValueError(
                f"Expected at least {4*nJ} columns (got {X.shape[1]}). "
                "Columns must be [xmean0,ymean0,...,xmean32,ymean32, xstd0,ystd0,...,xstd32,ystd32]."
            )

        # define output names (order matches transform stacking)
        hand_cols = self._hand_feature_names()
        diff_cols = self._diff_feature_names()
        knee_cols = self._knee_std_cols()
        std_cols  = ['left_std', 'right_std']
        others    = ['torso_length']
        self.output_features_ = hand_cols + diff_cols + knee_cols + std_cols + others
        return self

    def transform(self, X):
        if X.ndim != 2:
            raise ValueError("X must be 2D NumPy array.")

        # views by component (shape: n_samples x n_joints)
        Xm = self._view_comp(X, 'xmean')
        Ym = self._view_comp(X, 'ymean')
        Xs = self._view_comp(X, 'xstd')
        Ys = self._view_comp(X, 'ystd')

        # 1) Hand averages
        def hand_avg(comp_mat, idxs):
            return comp_mat[:, idxs].mean(axis=1)

        hand_feats = []
        for key, idxs in (('left', self.keypoints_hands['left']),
                          ('right', self.keypoints_hands['right'])):
            hand_feats.extend([
                hand_avg(Xm, idxs),  # xmean{key}_hand
                hand_avg(Ym, idxs),  # ymean{key}_hand
                hand_avg(Xs, idxs),  # xstd{key}_hand
                hand_avg(Ys, idxs),  # ystd{key}_hand
            ])

        hand_dict = {
            'left':  {'xmean': hand_feats[0], 'ymean': hand_feats[1], 'xstd': hand_feats[2], 'ystd': hand_feats[3]},
            'right': {'xmean': hand_feats[4], 'ymean': hand_feats[5], 'xstd': hand_feats[6], 'ystd': hand_feats[7]},
        }

        # 2) Torso length (avg of the two diagonals): joints (11,23) and (12,24)
        left_torso  = np.sqrt((Xm[:,11] - Xm[:,23])**2 + (Ym[:,11] - Ym[:,23])**2)
        right_torso = np.sqrt((Xm[:,12] - Xm[:,24])**2 + (Ym[:,12] - Ym[:,24])**2)
        torso_length = (left_torso + right_torso) / 2.0
        denom = np.where(torso_length == 0.0, np.nan, torso_length)

        # 3) Diffs (xmean/ymean), normalized by torso_length
        def ref_values(comp, ref):
            if isinstance(ref, str):
                if ref == 'left_hand':
                    return hand_dict['left'][comp]
                elif ref == 'right_hand':
                    return hand_dict['right'][comp]
                else:
                    raise ValueError(f"Unknown ref '{ref}'")
            # int joint index
            return (Xm if comp == 'xmean' else Ym)[:, ref]

        diff_feats = []
        for a, b in self.diff_pairs:
            for comp in self.components_diff:
                va = ref_values(comp, a)
                vb = ref_values(comp, b)
                diff_feats.append((va - vb) / denom)

        # 4) Aggregate stds over left/right sides
        left_std  = Xs[:, self.keypoint_side['left']].sum(axis=1) + Ys[:, self.keypoint_side['left']].sum(axis=1)
        right_std = Xs[:, self.keypoint_side['right']].sum(axis=1) + Ys[:, self.keypoint_side['right']].sum(axis=1)

        # 5) Knee std passthrough (xstd25, ystd25, xstd26, ystd26)
        knee_std = [Xs[:,25], Ys[:,25], Xs[:,26], Ys[:,26]]

        # Stack in the documented order
        out_blocks = hand_feats + diff_feats + knee_std + [left_std, right_std] + [torso_length]
        return np.column_stack(out_blocks).astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_features_, dtype=object)#metrics