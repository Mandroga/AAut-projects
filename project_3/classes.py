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
