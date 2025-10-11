# %% imports
%run imports.py
  
# %% data
with open("Xtrain1.pkl", "rb") as f:
    X = pickle.load(f)
Y= np.load("Ytrain1.npy")
df = pd.DataFrame(X)
df['target'] = Y

# %% df_ - df unpacked
df_ = df.copy()

x_mean_i = [i for i in range(0, 66, 2)]
y_mean_i = [i for i in range(1, 66, 2)]
x_std_i = [i for i in range(66, 132, 2)]
y_std_i = [i for i in range(67, 132, 2)]
keypoints = list(range(33))
for i in keypoints:
    df_[f'xmean{i}'] = df_['Skeleton_Features'].apply(lambda x: x[x_mean_i[i]])
    df_[f'ymean{i}'] = df_['Skeleton_Features'].apply(lambda x: x[y_mean_i[i]])
    df_[f'xstd{i}'] = df_['Skeleton_Features'].apply(lambda x: x[x_std_i[i]])
    df_[f'ystd{i}'] = df_['Skeleton_Features'].apply(lambda x: x[y_std_i[i]])
df_ = df_.drop(['Skeleton_Features'], axis=1)
print(df_.groupby('Patient_Id')['target'].value_counts().unstack())

# %% keypoint parts
keypoint_side = {'left':[4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
                 'right':[1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31]
                }
keypoints_hands = {'left': [15,17,19,21], 'right': [16,18,20,22]}
def make_cols(indexes, components=['xmean','ymean','xstd','ystd']):
    return [txt + str(i) for i in indexes for txt in components]
side_cols = {k: make_cols(v) for k,v in keypoint_side.items()}

# %% df processed
df_processed = df_.copy()

#impairment side
if 1:
    df_stroke = df_processed.copy()
    for patient_id in range(1,15):
        for key, indexes in keypoint_side.items():
            cols = [txt + str(i) for txt in ['xstd', 'ystd'] for i in indexes]
            mask = df_stroke['Patient_Id']==patient_id
            df_stroke.loc[mask, key + 'std'] = df_stroke[mask][cols].sum().sum()
    df_stroke['impairment_side'] = (df_stroke['leftstd'] > df_stroke['rightstd']).astype(int)
    df_processed['impairment_side'] = df_stroke['impairment_side']

# average hand keypoints for hand feature!
if 1:
    hand_cols = []
    for key, indexes in keypoints_hands.items():
        cols = make_cols(indexes)
        components = ['xmean','ymean','xstd','ystd']
        for component in components:
            sub_cols = [col for col in cols if component in col]
            col_name = f'{component}{key}_hand'
            df_processed[col_name] = df_processed[sub_cols].mean(axis=1)
            hand_cols.append(col_name)
    
# diff cols 
if 1:
    diff_cols = []
    diff_index_list = [(25,23), (26,24), ('left_hand', 9), ('right_hand',10)]
    components = ['xmean','ymean']
    for di1,di2 in diff_index_list:
        for component in components:
            col_name = f'{component}diff{di1}-{di2}'
            df_processed[col_name] = df_processed[f'{component}{di1}'] - df_processed[f'{component}{di2}']
            diff_cols.append(col_name)

# normalize knee std and hand std by torso lenght!
if 1:
    left_torso = np.sqrt((df_['xmean11'] - df_['xmean23'])**2 + (df_['ymean11'] - df_['ymean23'])**2)
    right_torso = np.sqrt((df_['xmean12'] - df_['xmean24'])**2 + (df_['ymean12'] - df_['ymean24'])**2)
    torso_length = (left_torso + right_torso) / 2
   # df_processed['torso_length'] = torso_length

knee_std_cols = [c+str(i) for i in [25,26] for c in ['xstd','ystd']]

df_processed = df_processed[['Patient_Id'] + hand_cols + diff_cols + knee_std_cols + ['impairment_side','target']].copy()

#weights
if 1:
    w = df_processed[[txt+str(j) for txt in ['xstd','ystd'] for j in ['left_hand','right_hand',25,26]]].copy()
    scaler = MinMaxScaler()
    w.div(torso_length, axis=0)
    w = scaler.fit_transform(w.values)
    w =  w.sum(axis=1)
    w *= w
    w = w / w.max()
    


