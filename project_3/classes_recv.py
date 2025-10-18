#%%w
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
#%%

class StringtoOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = ['E1','E2','E3','E4','E5']
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):    
        X_transformed = X.drop(columns='Exercise_Id').reset_index(drop=True)
        E1, E2, E3, E4, E5 = [], [], [], [], []
        for row in X['Exercise_Id']:
            if row == 'E1':
                E1.append(1)
                E2.append(0)
                E3.append(0)
                E4.append(0)
                E5.append(0)
            elif row == 'E2':
                E1.append(0)
                E2.append(1)
                E3.append(0)
                E4.append(0)
                E5.append(0)
            elif row == 'E3':
                E1.append(0)
                E2.append(0)
                E3.append(1)
                E4.append(0)
                E5.append(0)
            elif row == 'E4':
                E1.append(0)
                E2.append(0)
                E3.append(0)
                E4.append(1)
                E5.append(0)
            elif row == 'E5':
                E1.append(0)
                E2.append(0)
                E3.append(0)
                E4.append(0)
                E5.append(1)
            else:
                raise ValueError(f"Unexpected Exercise_Id value: {row}")
        X_transformed['E1'] = np.array(E1)
        X_transformed['E2'] = np.array(E2)
        X_transformed['E3'] = np.array(E3)
        X_transformed['E4'] = np.array(E4)
        X_transformed['E5'] = E5
        return X_transformed
#%%

class Distances(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    @staticmethod
    def sum_consecutive_distances(points):
        """
        points: array-like of shape (T, D) where D is 1 (e.g. x only) or >1 (e.g. x,y).
        Returns the sum of Euclidean distances between consecutive points:
            sum_{t=0..T-2} ||points[t+1] - points[t]||.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        if pts.shape[0] < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)            # shape (T-1, D)
        if pts.shape[1] == 1:
            return float(np.sum(np.abs(diffs[:, 0])))   # 1D -> absolute diffs
        return float(np.sum(np.linalg.norm(diffs, axis=1)))  # Euclidean per step
    
    def transform(self, X_onehot):
        X_transformed = []
        for i in range(len(14)):
            X_transformed.append(i+1)
        for row in X_onehot.itertuples():
            Patient_Id = row['Patient_Id']
            ss= row['Skeleton_Sequence']
            Patient_arr = X_transformed[Patient_Id-1]
            e1 = row['E1']
            e2 = row['E2']      
            e3 = row['E3']
            e4 = row['E4']
            e5 = row['E5']
            Patient_arr.append([[e1,e2,e3,e4,e5],])
            if e1 == 1 or e2 == 1 or e3 == 1 or e4 == 1:    
                rpinky = ss[:,17*2:17*2+1+1]
                lpinky = ss[:,18*2:18*2+1+1]
                distrpinky = self.sum_consecutive_distances(rpinky)
                distlpinky = self.sum_consecutive_distances(lpinky)
                Patient_arr[1].append([distrpinky, distlpinky])
            if e4 == 1 or e5 == 1:
                rtoe = ss[:,31*2:31*2+1+1]
                ltoe = ss[:,32*2:32*2+1+1]
     
                distrtoe = self.sum_consecutive_distances(rtoe)
                distltoe = self.sum_consecutive_distances(ltoe)
                lknee = ss[:,26*2:26*2+1+1]
                rknee = ss[:,27*2:27*2+1+1]
                distlknee = self.sum_consecutive_distances(lknee)
                distrknee = self.sum_consecutive_distances(rknee)
                if len(Patient_arr[1])==1:
                    Patient_arr[1].append([distrtoe, distltoe, distrknee, distlknee])
                else:
                    Patient_arr[1][1].append(distrtoe, distltoe, distrknee, distlknee)
        return X_transformed
# %%

class Velocities(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    @staticmethod
    def average_velocity(points, fps=30):
        """
        points: array-like of shape (T, D) where D is 1 (e.g. x only) or >1 (e.g. x,y).
        Returns the sum of Euclidean velocities between consecutive points:
            sum_{t=0..T-2} ||points[t+1] - points[t]|| * fps.
        """
        pts = np.asarray(points, dtype=float)
        num_frames = pts.shape[0]
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        if pts.shape[0] < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)            # shape (T-1, D)
        if pts.shape[1] == 1:
            return float(np.sum(np.abs(diffs[:, 0])) * fps / num_frames)   # 1D -> absolute diffs
        return float(np.sum(np.linalg.norm(diffs, axis=1)) * fps /num_frames)  # Euclidean per step
    
    def transform(self, X_onehot):
        X_transformed = []
        for i in range(len(14)):
            X_transformed.append([i+1,])
        for row in X_onehot.itertuples():
            Patient_Id = row['Patient_Id']
            ss= row['Skeleton_Sequence']
            Patient_arr = X_transformed[Patient_Id-1]
            e1 = row['E1']
            e2 = row['E2']      
            e3 = row['E3']
            e4 = row['E4']
            e5 = row['E5']
            Patient_arr.append([[e1,e2,e3,e4,e5],])
            if e1 == 1 or e2 == 1 or e3 == 1 or e4 == 1:    
                rpinky = ss[:,17*2:17*2+1+1]
                lpinky = ss[:,18*2:18*2+1+1]
                velrpinky = self.average_velocity(rpinky)
                vellpinky = self.average_velocity(lpinky)
                Patient_arr[1].append([velrpinky,vellpinky])
            if e4 == 1 or e5 == 1:
                rtoe = ss[:,31*2:31*2+1+1]
                ltoe = ss[:,32*2:32*2+1+1]
     
                velrtoe = self.average_velocity(rtoe)
                velltoe = self.average_velocity(ltoe)
                lknee = ss[:,26*2:26*2+1+1]
                rknee = ss[:,27*2:27*2+1+1]
                vellknee = self.average_velocity(lknee)
                velrknee = self.average_velocity(rknee)
                if len(Patient_arr[1])==1:
                    Patient_arr[1].append([velrtoe, velltoe, velrknee, vellknee])
                else:
                    Patient_arr[1][1].append(velrtoe, velltoe, velrknee, vellknee)
        return X_transformed
    
class Amplitude(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    @staticmethod
    def amplitude(points):
        """
        points: array-like of shape (T, D) where D is 1 (e.g. x only) or >1 (e.g. x,y).
        Returns the amplitude (max - min) for each dimension, summed:
            sum_{d=1..D} (max(points[:,d]) - min(points[:,d])).
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        if pts.shape[0] < 1:
            return 0.0
        return float(np.sum(np.ptp(pts, axis=0)))  # ptp = max - min per dimension
    
    def transform(self, X_onehot):
        X_transformed = []
        for i in range(len(14)):
            X_transformed.append([i+1,])
        for row in X_onehot.itertuples():
            Patient_Id = row['Patient_Id']
            ss= row['Skeleton_Sequence']
            Patient_arr = X_transformed[Patient_Id-1]
            e1 = row['E1']
            e2 = row['E2']      
            e3 = row['E3']
            e4 = row['E4']
            e5 = row['E5']
            Patient_arr.append([[e1,e2,e3,e4,e5],])
            if e1 == 1 or e2 == 1 or e3 == 1 or e4 == 1:    
                rpinky = ss[:,17*2:17*2+1+1]
                lpinky = ss[:,18*2:18*2+1+1]
                amprpinky = self.amplitude(rpinky)
                amplpinky = self.amplitude(lpinky)
                Patient_arr[1].append([amprpinky,amplpinky])
            if e4 == 1 or e5 == 1:
                rtoe = ss[:,31*2:31*2+1+1]
                ltoe = ss[:,32*2:32*2+1+1]
     
                amprtoe = self.amplitude(rtoe)
                ampltoe = self.amplitude(ltoe)
                lknee = ss[:,26*2:26*2+1+1]
                rknee = ss[:,27*2:27*2+1+1]
                amplknee = self.amplitude(lknee)
                amprknee = self.amplitude(rknee)
                if len(Patient_arr[1])==1:
                    Patient_arr[1].append([amprtoe, ampltoe, amprknee, amplknee])
                else:
                    Patient_arr[1][1].append(amprtoe, ampltoe, amprknee, amplknee)
        return X_transformed


#%%
class FeatureEngMerge(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.string_encoder = StringtoOneHotEncoder()
        #self.distance_extractor = Distances()
        #self.velocity_extractor = Velocities()
        #self.amplitude_extractor = Amplitude()
    
    def fit(self, X, y=None):
        return self
    
    @staticmethod
    def sum_consecutive_distances(points):
        """
        points: array-like of shape (T, D) where D is 1 (e.g. x only) or >1 (e.g. x,y).
        Returns the sum of Euclidean distances between consecutive points:
            sum_{t=0..T-2} ||points[t+1] - points[t]||.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        if pts.shape[0] < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)            # shape (T-1, D)
        if pts.shape[1] == 1:
            return float(np.sum(np.abs(diffs[:, 0])))   # 1D -> absolute diffs
        return float(np.sum(np.linalg.norm(diffs, axis=1)))  # Euclidean per step

    @staticmethod
    def average_velocity(points, fps=1):
        """
        points: array-like of shape (T, D) where D is 1 (e.g. x only) or >1 (e.g. x,y).
        Returns the sum of Euclidean velocities between consecutive points:
            sum_{t=0..T-2} ||points[t+1] - points[t]|| * fps.
        """
        pts = np.asarray(points, dtype=float)
        num_frames = pts.shape[0]
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        if pts.shape[0] < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)            # shape (T-1, D)
        if pts.shape[1] == 1:
            return float(np.sum(np.abs(diffs[:, 0])) * fps / num_frames)   # 1D -> absolute diffs
        return float(np.sum(np.linalg.norm(diffs, axis=1)) * fps /num_frames)  # Euclidean per step
    
    @staticmethod
    def amplitude(points):
        """
        points: array-like of shape (T, D) where D is 1 (e.g. x only) or >1 (e.g. x,y).
        Returns the amplitude (max - min) for each dimension, summed:
            sum_{d=1..D} (max(points[:,d]) - min(points[:,d])).
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        if pts.shape[0] < 1:
            return 0.0
        return np.ptp(pts, axis=0)  # ptp = max - min per dimension
    
    @staticmethod
    def orientationchange(points):
        """
        points: array-like of shape (T, D) where D is 1 (e.g. x only) or >1 (e.g. x,y).
        Returns the number of times the orientation of the movement vector changes significantly.
        A significant change is defined as a change in angle greater than 45 degrees.:
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 1)
        if pts.shape[0] < 2:
            return 0.0
        vectors = np.diff(pts, axis=0)            # shape (T-1, D)
        orientations = []
        for vec in vectors:
            norm = np.linalg.norm(vec)
            if norm == 0:
                orientations.append(0)
            else:
                unit_vec = vec / norm
                angle = np.arctan2(unit_vec[1], unit_vec[0])  # angle in radians
                orientations.append(angle)
        orientation_changes = 0
        for i in range(1, len(orientations)):
            angle_diff = np.abs(orientations[i] - orientations[i-1])
            if angle_diff > np.pi / 4:  # greater than 45 degrees
                orientation_changes += 1
        return orientation_changes


    def transform(self, X):
        X_onehot = self.string_encoder.transform(X)
        X_transformed = []
        for i in range(14):
            X_transformed.append([i+1,])
        for row in X_onehot.itertuples():
            Patient_Id = row.Patient_Id
            ss = row.Skeleton_Sequence
            Patient_arr = X_transformed[Patient_Id-1]
            e1 = row.E1
            e2 = row.E2
            e3 = row.E3
            e4 = row.E4
            e5 = row.E5

            current_features =[]          
            
            if e3 == 1 or e4 == 1:
                


                rpinky = ss[:,17*2:17*2+1+1]
                lpinky = ss[:,18*2:18*2+1+1]

                velrpinky = self.average_velocity(rpinky)
                vellpinky = self.average_velocity(lpinky)
                current_features.append(velrpinky)
                current_features.append(vellpinky)

                amprpinky = self.amplitude(rpinky)
                amplpinky = self.amplitude(lpinky)
                current_features.append(amprpinky[1])
                current_features.append(amplpinky[1])
                if e3 == 1:
                    rindex = ss[:,19*2:19*2+1+1]
                    lindex = ss[:,20*2:20*2+1+1]
                    distance_pinky_index_r = np.linalg.norm(rpinky - rindex)
                    distance_pinky_index_l = np.linalg.norm(lpinky - lindex)
                    current_features.append(distance_pinky_index_r)
                    current_features.append(distance_pinky_index_l)

                    rwrist = ss[:,16*2:16*2+1+1]
                    lwrist = ss[:,15*2:15*2+1+1]
                    velrwrist = self.average_velocity(rwrist)
                    vellwrist = self.average_velocity(lwrist)
                    current_features.append(velrwrist)
                    current_features.append(vellwrist)

                    Patient_arr.append([[e3,e4,e5],current_features])

                if e4 == 1:
                    distrpinky = self.sum_consecutive_distances(rpinky)
                    distlpinky = self.sum_consecutive_distances(lpinky)
                    normalized_distrpinky = distrpinky / (2 * amprpinky[1]) if amprpinky[1] != 0 else 0
                    normalized_distlpinky = distlpinky / (2 * amplpinky[1]) if amplpinky[1] != 0 else 0
                    current_features.append(normalized_distrpinky)
                    current_features.append(normalized_distlpinky)

                    Patient_arr.append([[e3,e4,e5],current_features])
            if e5 == 1:

                #rtoe = ss[:,31*2:31*2+1+1]
                #ltoe = ss[:,32*2:32*2+1+1]
     
                #amprtoe = self.amplitude(rtoe)
                #ampltoe = self.amplitude(ltoe)
                #current_features.append(amprtoe, ampltoe)

                lknee = ss[:,26*2:26*2+1+1]
                rknee = ss[:,27*2:27*2+1+1]
                amplknee = self.amplitude(lknee)
                amprknee = self.amplitude(rknee)
                current_features.append(amprknee[1])
                current_features.append(amplknee[1])

                distlknee = self.sum_consecutive_distances(lknee)
                distrknee = self.sum_consecutive_distances(rknee)
                normalized_distlknee = distlknee / (2 * amplknee[1]) if amplknee[1] != 0 else 0
                normalized_distrknee = distrknee / (2 * amprknee[1]) if amprknee[1] != 0 else 0
                current_features.append(normalized_distrknee)
                current_features.append(normalized_distlknee)

                vellknee = self.average_velocity(lknee)
                velrknee = self.average_velocity(rknee)
                current_features.append(velrknee)
                current_features.append(vellknee)

                orchlnknee = self.orientationchange(lknee)
                orchrknee = self.orientationchange(rknee)
                current_features.append(orchrknee)
                current_features.append(orchlnknee)

                Patient_arr.append([[e3, e4, e5],current_features])
        return np.array(X_transformed, dtype=object)

        
        #%%
        
        
        
        
        
        
        
        
        
        X_distances = self.distance_extractor.transform(X_onehot)
        X_velocities = self.velocity_extractor.transform(X_onehot)
        X_amplitudes = self.amplitude_extractor.transform(X_onehot)
        
        # Merge features
        X_transformed = []
        for i in range(len(X_distances)):
            patient_id = X_distances[i][0]
            features = {}
            # Add distance features
            for j, val in enumerate(X_distances[i][1][1]):
                features[f'distance_feature_{j+1}'] = val
            # Add velocity features
            for j, val in enumerate(X_velocities[i][1][1]):
                features[f'velocity_feature_{j+1}'] = val
            # Add amplitude features
            for j, val in enumerate(X_amplitudes[i][1][1]):
                features[f'amplitude_feature_{j+1}'] = val
            
            X_transformed.append({
                "Patient_Id": patient_id,
                **features
            })
        
        df = pd.DataFrame(X_transformed)
        df.fillna(0, inplace=True)  # Handle any NaNs from feature extraction
        return df
# %%
import numpy as np
import pandas as pd

# ---- low-level helpers ----
def ensure_2d_pts(points):
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 1)
    return pts

def sum_consecutive_distances(points):
    pts = ensure_2d_pts(points)
    if pts.shape[0] < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1) if pts.shape[1] > 1 else np.abs(diffs[:,0])
    return float(np.sum(dists))

def average_velocity(points, fps=1.0):
    pts = ensure_2d_pts(points)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1) if pts.shape[1] > 1 else np.abs(diffs[:,0])
    return float(np.mean(step_dists) * fps)

def amplitude_xy(points):
    """
    Returns (amp_x, amp_y) for 2D coordinates.
    If points are 1D returns (amp, 0.0)
    """
    pts = ensure_2d_pts(points)
    if pts.shape[0] < 1:
        return (0.0, 0.0)
    ptp = np.ptp(pts, axis=0)
    if ptp.size == 1:
        return (float(ptp[0]), 0.0)
    # If more dims, take first two as x,y (safe if data is (T,2))
    x = float(ptp[0])
    y = float(ptp[1]) if ptp.size > 1 else 0.0
    return (x, y)

def direction_change_count(points, angle_thresh_rad=0.5):
    """
    Count how many times direction changes by more than angle_thresh_rad.
    For 1D data uses sign-change count of diffs.
    """
    pts = ensure_2d_pts(points)
    if pts.shape[0] < 3:  # need at least 3 frames to have two steps to compare
        return 0
    diffs = np.diff(pts, axis=0)
    if pts.shape[1] == 1:
        # sign changes of derivative
        signs = np.sign(diffs[:,0])
        # count sign flips excluding zeros
        nonzero = signs != 0
        filtered = signs[nonzero]
        if filtered.size < 2:
            return 0
        return int(np.sum(filtered[1:] != filtered[:-1]))
    else:
        angles = np.arctan2(diffs[:,1], diffs[:,0])  # per-step angle
        dangles = np.abs(np.diff(angles))
        # unwrap to smallest difference
        dangles = np.minimum(dangles, 2*np.pi - dangles)
        return int(np.sum(dangles > angle_thresh_rad))

def mean_distance_between(seqa, seqb):
    a = ensure_2d_pts(seqa)
    b = ensure_2d_pts(seqb)
    # if different lengths, truncate to min length
    m = min(a.shape[0], b.shape[0])
    if m == 0:
        return 0.0
    a = a[:m]
    b = b[:m]
    dists = np.linalg.norm(a - b, axis=1) if a.shape[1] > 1 or b.shape[1] > 1 else np.abs(a[:,0] - b[:,0])
    return float(np.mean(dists))

# ---- per-row (exercise entry) feature extraction ----
def features_from_row(ss, fps=1.0):
    """
    Given skeleton sequence ss (T, C), compute the scalar features for
    pinky and knee and average pinky-index distance.
    Returns dictionary with keys for exercises that might use them.
    The function does not check which exercise is active; caller will use appropriate fields.
    """
    # joint indices 
    # rpinky joint index = 17, lpinky = 18
    # rindex = 19, lindex = 20
    # rwrist = 16, lwrist = 15
    # lknee = 26, rknee = 27
    idx = lambda j: (j*2, j*2 + 2)

    def slice_joint(j):
        a,b = idx(j)
        return ss[:, a:b]  # shape (T,2) usually

    out = {}
    # Pinky (right and left)
    rpinky = slice_joint(17)
    lpinky = slice_joint(18)
    rindex = slice_joint(19)
    lindex = slice_joint(20)
    rwrist = slice_joint(16)
    lwrist = slice_joint(15)
    # Knees
    lknee = slice_joint(26)
    rknee = slice_joint(27)

    # pinky features (scalars)
    out['pinky_vel_mean_r'] = average_velocity(rpinky, fps=fps)
    out['pinky_vel_mean_l'] = average_velocity(lpinky, fps=fps)
    out['pinky_total_dist_r'] = sum_consecutive_distances(rpinky)
    out['pinky_total_dist_l'] = sum_consecutive_distances(lpinky)
    amp_r_x, amp_r_y = amplitude_xy(rpinky)
    amp_l_x, amp_l_y = amplitude_xy(lpinky)
    out['pinky_amp_r_x'] = amp_r_x
    out['pinky_amp_r_y'] = amp_r_y
    out['pinky_amp_l_x'] = amp_l_x
    out['pinky_amp_l_y'] = amp_l_y
    # direction change counts
    out['pinky_dirchanges_r'] = direction_change_count(rpinky)
    out['pinky_dirchanges_l'] = direction_change_count(lpinky)
    # mean distance index<->pinky (per-side)
    out['mean_dist_pinky_index_r'] = mean_distance_between(rpinky, rindex)
    out['mean_dist_pinky_index_l'] = mean_distance_between(lpinky, lindex)
    # wrist velocities (if needed)
    #out['rwrist_vel'] = average_velocity(rwrist, fps=fps)
    #out['lwrist_vel'] = average_velocity(lwrist, fps=fps)

    # knee features (scalars)
    out['knee_amp_r_x'], out['knee_amp_r_y'] = amplitude_xy(rknee)
    out['knee_amp_l_x'], out['knee_amp_l_y'] = amplitude_xy(lknee)
    out['knee_total_dist_r'] = sum_consecutive_distances(rknee)
    out['knee_total_dist_l'] = sum_consecutive_distances(lknee)
    out['knee_vel_mean_r'] = average_velocity(rknee, fps=fps)
    out['knee_vel_mean_l'] = average_velocity(lknee, fps=fps)
    out['knee_dirchanges_r'] = direction_change_count(rknee)
    out['knee_dirchanges_l'] = direction_change_count(lknee)

    # normalized distances optional: normalized by (amp_x + amp_y) per side, try to avoid zero division
    denom_r_pinky = ( amp_r_y) if ( amp_r_y) != 0 else 1.0
    denom_l_pinky = ( amp_l_y) if ( amp_l_y) != 0 else 1.0
    out['pinky_total_dist_r_norm'] = out['pinky_total_dist_r'] / denom_r_pinky
    out['pinky_total_dist_l_norm'] = out['pinky_total_dist_l'] / denom_l_pinky

    denom_r_knee = ( out['knee_amp_r_y']) if (out['knee_amp_r_x'] + out['knee_amp_r_y']) != 0 else 1.0
    denom_l_knee = ( out['knee_amp_l_y']) if (out['knee_amp_l_x'] + out['knee_amp_l_y']) != 0 else 1.0
    out['knee_total_dist_r_norm'] = out['knee_total_dist_r'] / denom_r_knee
    out['knee_total_dist_l_norm'] = out['knee_total_dist_l'] / denom_l_knee

    return out

# ---- per-patient aggregation ----
def build_patient_feature_matrix(df, fps=1.0, n_exercises=5, fill_value=0.0, return_dataframe=True):
    """
    df: pandas DataFrame with columns Patient_Id, Skeleton_Sequence, E1..E5 (binary)
    Returns:
      - if return_dataframe True: pandas.DataFrame with named columns (one row per patient)
      - else: numpy ndarray (n_patients, P) and list of column names
    Final columns:
      patient_id,
      presence_E1..E5 (0/1),
      count_E1..E5,
      mean features for E1, mean features for E2, ... (in a fixed order; see column names)
    """
    # define the per-entry feature names in fixed order
    sample_feats = features_from_row(np.zeros((3, 60)))  # just to extract keys; adjust 60 to be >= largest joint index*2
    feat_keys = list(sample_feats.keys())

    # helper to get exercise index from events list (assumes one-hot vector length n_exercises)
    def ex_index_from_events(events):
        try:
            arr = np.asarray(events, dtype=int)
            if arr.size >= n_exercises:
                return int(np.argmax(arr[:n_exercises]))
            # fallback: if events are list of flags longer/shorter, use argmax
            return int(np.argmax(arr))
        except Exception:
            return 0

    # collect per-patient per-exercise feature lists
    patients = {}
    for _, row in df.iterrows():
        pid = int(row['Patient_Id'])
        ss = np.asarray(row['Skeleton_Sequence'], dtype=float)
        events = [int(row.get(f"E{i}", 0)) for i in range(1, n_exercises+1)]
        ex_idx = ex_index_from_events(events)
        entry_feats = features_from_row(ss, fps=fps)
        feat_vector = [entry_feats[k] for k in feat_keys]

        if pid not in patients:
            patients[pid] = { 'per_ex': [ [] for _ in range(n_exercises) ], 'presence': np.zeros(n_exercises, dtype=int) }
        patients[pid]['per_ex'][ex_idx].append(feat_vector)
        patients[pid]['presence'][ex_idx] = 1

    # build rows
    pid_list = sorted(patients.keys())
    rows = []
    col_names = []

    # header: patient_id, presence_E1..E5, count_E1..E5
    col_names.append('patient_id')
    for i in range(1, n_exercises+1):
        col_names.append(f'presence_E{i}')
    for i in range(1, n_exercises+1):
        col_names.append(f'count_E{i}')

    # then feature column names: for each exercise, for each feat_key
    for i in range(1, n_exercises+1):
        for k in feat_keys:
            col_names.append(f'E{i}_{k}')

    for pid in pid_list:
        info = patients[pid]
        presence = info['presence']
        counts = np.array([len(info['per_ex'][j]) for j in range(n_exercises)], dtype=float)
        # compute mean feature vector per exercise (if no entries -> fill fill_value)
        mean_feats_per_ex = []
        for j in range(n_exercises):
            lst = info['per_ex'][j]
            if len(lst) == 0:
                mean_vec = np.full(len(feat_keys), fill_value, dtype=float)
            else:
                arr = np.vstack(lst)  # (n_entries, n_feats)
                mean_vec = np.nanmean(arr, axis=0)
                mean_vec = np.where(np.isnan(mean_vec), fill_value, mean_vec)
            mean_feats_per_ex.append(mean_vec)

        # flatten
        mean_feats_flat = np.concatenate(mean_feats_per_ex)
        row = np.concatenate([[int(pid)], presence.astype(float), counts.astype(float), mean_feats_flat])
        rows.append(row)

    if len(rows) == 0:
        # no patients -> empty frame
        df_out = pd.DataFrame(columns=col_names)
        return df_out if return_dataframe else (np.empty((0, len(col_names))), col_names)

    mat = np.vstack(rows)
    if return_dataframe:
        df_out = pd.DataFrame(mat, columns=col_names)
        # cast patient_id back to int
        df_out['patient_id'] = df_out['patient_id'].astype(int)
        # presence back to int
        for i in range(1, n_exercises+1):
            df_out[f'presence_E{i}'] = df_out[f'presence_E{i}'].astype(int)
        return df_out
    else:
        return mat, col_names


#%%

class FeatEngMerge2(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.string_encoder = StringtoOneHotEncoder()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Transform input X (raw dataframe) into a fixed-size numpy array per patient.

        Assumptions:
          - self.string_encoder.transform(X) returns a pandas DataFrame with columns:
              'Patient_Id', 'Skeleton_Sequence', 'E1','E2','E3','E4','E5'
          - The helper functions you pasted are defined at module scope:
              features_from_row(...), build_patient_feature_matrix(...)
        Returns:
          - numpy.ndarray of shape (n_patients, P) where P is fixed.
        Side-effect:
          - sets self.feature_names_ to the list of column names (order of columns).
        """
        # 1) run the string / one-hot encoder you already have
        X_onehot = self.string_encoder.transform(X)

        # 2) Use the aggregation helper to compute per-patient fixed-length features.
        #    build_patient_feature_matrix returns (mat, col_names) when return_dataframe=False
        try:
            mat, col_names = build_patient_feature_matrix(X_onehot,
                                                         fps=1.0,
                                                         n_exercises=5,
                                                         fill_value=0.0,
                                                         return_dataframe=False)
        except TypeError:
            # If your helper returns only the DataFrame by default, call with return_dataframe=True then convert:
            df_out = build_patient_feature_matrix(X_onehot,
                                                 fps=1.0,
                                                 n_exercises=5,
                                                 fill_value=0.0,
                                                 return_dataframe=True)
            col_names = list(df_out.columns)
            mat = df_out.values.astype(float)

        # Save column names for later (helpful for pipelines / interpretation)
        self.feature_names_ = col_names

        # mat is the final fixed-size feature matrix (one row per patient)
        return mat

# %%




class FeatEngMerge3(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.string_encoder = StringtoOneHotEncoder()
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform exercise-level DataFrame X -> patient-level numeric array X_np.
        Returns:
        X_np: np.ndarray shape (n_patients, n_features) dtype float
        Side-effects:
        self.feature_names_ = list of feature names
        self.patient_ids_ = np.ndarray of patient ids corresponding to rows
        """
        from collections import defaultdict
        X_onehot = self.string_encoder.transform(X)
        X = X_onehot  # for clarity
        # --- local helpers ---
        def ensure_2d_pts(points):
            pts = np.asarray(points, dtype=float)
            if pts.ndim == 1:
                pts = pts.reshape(-1, 1)
            return pts

        def avg_velocity(points, fps=1.0):
            pts = ensure_2d_pts(points)
            if pts.shape[0] < 2:
                return 0.0
            diffs = np.diff(pts, axis=0)
            if pts.shape[1] == 1:
                step_dists = np.abs(diffs[:, 0])
            else:
                step_dists = np.linalg.norm(diffs, axis=1)
            return float(np.mean(step_dists) * fps)

        def sum_consec(points):
            pts = ensure_2d_pts(points)
            if pts.shape[0] < 2:
                return 0.0
            diffs = np.diff(pts, axis=0)
            if pts.shape[1] == 1:
                dists = np.abs(diffs[:, 0])
            else:
                dists = np.linalg.norm(diffs, axis=1)
            return float(np.sum(dists))

        def amp_y(points):
            pts = ensure_2d_pts(points)
            if pts.shape[0] < 1:
                return 0.0
            # if 2D assume col 1 is y, otherwise use col 0
            return float(np.ptp(pts[:, 1]) if pts.shape[1] > 1 else np.ptp(pts[:, 0]))

        def joint_slice(ss, joint_idx):
            ss_arr = np.asarray(ss, dtype=float)
            if ss_arr.ndim == 1:
                # make it (T, C) with C >=1
                ss_arr = ss_arr.reshape(-1, max(1, ss_arr.size))
            a = int(joint_idx * 2)
            b = a + 2
            C = ss_arr.shape[1]
            if a >= C:
                # missing joint columns -> return zeros (T x min(2,C))
                return np.zeros((ss_arr.shape[0], min(2, C)))
            return ss_arr[:, a:min(b, C)]

        # --- containers per patient ---
        per_patient = defaultdict(lambda: defaultdict(list))
        ecols = ['E1', 'E2', 'E3', 'E4', 'E5']
        fps = getattr(self, "fps", 1.0)

        # Iterate rows (exercise-level)
        for _, row in X.iterrows():
            pid = int(row['Patient_Id'])
            ss = row['Skeleton_Sequence']
            events = [int(row.get(c, 0)) for c in ecols]
            ex_idx = int(np.argmax(events))  # 0..4: E1..E5

            # joints: pinky right=17, pinky left=18; knees right=27, left=26
            rpinky = joint_slice(ss, 17)
            lpinky = joint_slice(ss, 18)
            rknee = joint_slice(ss, 27)
            lknee = joint_slice(ss, 26)
            rindex = joint_slice(ss, 19)
            lindex = joint_slice(ss, 20)

            # pinky per-side
            vel_rp = avg_velocity(rpinky, fps=fps)
            vel_lp = avg_velocity(lpinky, fps=fps)
            dist_rp = sum_consec(rpinky)
            dist_lp = sum_consec(lpinky)
            amp_rp_y = amp_y(rpinky)
            amp_lp_y = amp_y(lpinky)

            # knee per-side
            vel_rk = avg_velocity(rknee, fps=fps)
            vel_lk = avg_velocity(lknee, fps=fps)
            dist_rk = sum_consec(rknee)
            dist_lk = sum_consec(lknee)
            amp_rk_y = amp_y(rknee)
            amp_lk_y = amp_y(lknee)

            # store per exercise index
            if ex_idx == 2:  # E3 -> pinky velocities + amp_y
                per_patient[pid]['pinky_vel_r_E3'].append(vel_rp)
                per_patient[pid]['pinky_vel_l_E3'].append(vel_lp)
                per_patient[pid]['pinky_amp_y_r_E3'].append(amp_rp_y)
                per_patient[pid]['pinky_amp_y_l_E3'].append(amp_lp_y)
            if ex_idx == 3:  # E4 -> pinky velocities, amp_y, dist
                per_patient[pid]['pinky_vel_r_E4'].append(vel_rp)
                per_patient[pid]['pinky_vel_l_E4'].append(vel_lp)
                per_patient[pid]['pinky_amp_y_r_E4'].append(amp_rp_y)
                per_patient[pid]['pinky_amp_y_l_E4'].append(amp_lp_y)
                per_patient[pid]['pinky_dist_r_E4'].append(dist_rp)
                per_patient[pid]['pinky_dist_l_E4'].append(dist_lp)
            if ex_idx == 4:  # E5 -> knee velocities, amp_y, dist
                per_patient[pid]['knee_vel_r_E5'].append(vel_rk)
                per_patient[pid]['knee_vel_l_E5'].append(vel_lk)
                per_patient[pid]['knee_amp_y_r_E5'].append(amp_rk_y)
                per_patient[pid]['knee_amp_y_l_E5'].append(amp_lk_y)
                per_patient[pid]['knee_dist_r_E5'].append(dist_rk)
                per_patient[pid]['knee_dist_l_E5'].append(dist_lk)

        # --- build output matrix ---
        feature_names = [
            # PINKY E3
            'mean_vel_pinky_r_E3', 'mean_vel_pinky_l_E3',
            'pinky_amp_y_r_E3', 'pinky_amp_y_l_E3',
            # PINKY E4
            'mean_vel_pinky_r_E4', 'mean_vel_pinky_l_E4',
            'pinky_amp_y_r_E4', 'pinky_amp_y_l_E4',
            'pinky_dist_norm_r_E4', 'pinky_dist_norm_l_E4',
            # KNEE E5
            'mean_vel_knee_r_E5', 'mean_vel_knee_l_E5',
            'knee_amp_y_r_E5', 'knee_amp_y_l_E5',
            'knee_dist_norm_r_E5', 'knee_dist_norm_l_E5'
        ]

        def safe_mean(lst):
            return float(np.mean(lst)) if len(lst) > 0 else 0.0

        patient_ids = sorted(per_patient.keys())
        rows = []
        for pid in patient_ids:
            d = per_patient[pid]
            # pinky E3
            mv_pr_E3 = safe_mean(d.get('pinky_vel_r_E3', []))
            mv_pl_E3 = safe_mean(d.get('pinky_vel_l_E3', []))
            pa_pr_E3 = safe_mean(d.get('pinky_amp_y_r_E3', []))
            pa_pl_E3 = safe_mean(d.get('pinky_amp_y_l_E3', []))
            # pinky E4
            mv_pr_E4 = safe_mean(d.get('pinky_vel_r_E4', []))
            mv_pl_E4 = safe_mean(d.get('pinky_vel_l_E4', []))
            pa_pr_E4 = safe_mean(d.get('pinky_amp_y_r_E4', []))
            pa_pl_E4 = safe_mean(d.get('pinky_amp_y_l_E4', []))
            dist_pr_E4 = safe_mean(d.get('pinky_dist_r_E4', []))
            dist_pl_E4 = safe_mean(d.get('pinky_dist_l_E4', []))
            denom_pr_p = 2.0 * pa_pr_E4 if pa_pr_E4 != 0 else np.nan
            denom_pl_p = 2.0 * pa_pl_E4 if pa_pl_E4 != 0 else np.nan
            pinky_dist_norm_r = float(dist_pr_E4 / denom_pr_p) if not np.isnan(denom_pr_p) else 0.0
            pinky_dist_norm_l = float(dist_pl_E4 / denom_pl_p) if not np.isnan(denom_pl_p) else 0.0

            # knee E5
            mv_kr_E5 = safe_mean(d.get('knee_vel_r_E5', []))
            mv_kl_E5 = safe_mean(d.get('knee_vel_l_E5', []))
            ka_kr_E5 = safe_mean(d.get('knee_amp_y_r_E5', []))
            ka_kl_E5 = safe_mean(d.get('knee_amp_y_l_E5', []))
            dist_kr_E5 = safe_mean(d.get('knee_dist_r_E5', []))
            dist_kl_E5 = safe_mean(d.get('knee_dist_l_E5', []))
            denom_kr_k = 2.0 * ka_kr_E5 if ka_kr_E5 != 0 else np.nan
            denom_kl_k = 2.0 * ka_kl_E5 if ka_kl_E5 != 0 else np.nan
            knee_dist_norm_r = float(dist_kr_E5 / denom_kr_k) if not np.isnan(denom_kr_k) else 0.0
            knee_dist_norm_l = float(dist_kl_E5 / denom_kl_k) if not np.isnan(denom_kl_k) else 0.0

            row = [
                mv_pr_E3 - mv_pl_E3, pa_pr_E3 - pa_pl_E3,
                mv_pr_E4 - mv_pl_E4, pa_pr_E4 - pa_pl_E4, pinky_dist_norm_r - pinky_dist_norm_l,
                mv_kr_E5 - mv_kl_E5, ka_kr_E5 - ka_kl_E5, knee_dist_norm_r - knee_dist_norm_l
            ]
            rows.append(row)

        X_np = np.asarray(rows, dtype=float)
        # store attributes for downstream use
        self.feature_names_ = feature_names
        self.patient_ids_ = np.asarray(patient_ids, dtype=int)

        return X_np

# %%
