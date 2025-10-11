# %% imports
%run data_imports.py

# %% df info
print(df.head(3))
print(df.info())
print(df.shape)
print(df['target'].unique())
print(df.iloc[0, 1].shape)
'''
df has 700 rows and 3 cols Patient id, skeleton features, target
target is categorical (exercise) values 0,1,2
each skeleton feature cell is an array (132,) 33 * 2 *2 (keypoints, (x,y), (mean,std))
'''
# %% df_
df_ = df.copy()

x_mean_i = [i for i in range(0, 66, 2)]
y_mean_i = [i for i in range(1, 66, 2)]
x_std_i = [i for i in range(66, 132, 2)]
y_std_i = [i for i in range(67, 132, 2)]
keypoints = list(range(33))
#nariz, pulsos, joelhos
#keypoints = [0,15,16, 25,26]
#keypoints = [0,15, 25]
for i in keypoints:
    df_[f'xmean{i}'] = df_['Skeleton_Features'].apply(lambda x: x[x_mean_i[i]])
    df_[f'ymean{i}'] = df_['Skeleton_Features'].apply(lambda x: -x[y_mean_i[i]])
    df_[f'xstd{i}'] = df_['Skeleton_Features'].apply(lambda x: x[x_std_i[i]])
    df_[f'ystd{i}'] = df_['Skeleton_Features'].apply(lambda x: x[y_std_i[i]])
df_ = df_.drop(['Skeleton_Features'], axis=1)


# %% target counts
#print(df['target'].value_counts())

print(df2.groupby('Patient_Id')['target'].value_counts().unstack())
print(df.groupby('Patient_Id')['target'].value_counts().unstack())
'''
Classes are unbalanced!
Each patient did a different number of execises of each kind
'''

# %% testing scores
n = 700
#indexes = list(range(33))
indexes = [0, 19,20, 15, 16, 21, 22, 25,26, 31, 32]
#indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
keypoint_cols = [txt+str(i) for i in indexes for txt in ['xmean','ymean','xstd','ystd']]
cols = keypoint_cols + ['Patient_Id']

def ex_score(ex_df_, i):
    ex_df = ex_df_.copy()
    #stdsc = StandardScaler()
    #ex_df[keypoint_cols] = stdsc.fit_transform(ex_df[keypoint_cols].values)
    if i == 0 or i == 1:
        ex_df['score'] = ex_df[[txt+str(j) for txt in ['xstd','ystd'] for j in [15,16,19,20, 21, 22]]].sum(axis=1)
    if i == 2:
        ex_df['score'] = ex_df[[txt+str(j) for txt in ['xstd','ystd'] for j in [25,26, 31, 32]]].sum(axis=1)
    return ex_df['score']

#n patients
if 0:
    ex_dfs = []
    for i in range(3):
        ex_df = df_[df_['target']==i].drop('target',axis=1).iloc[:n,:][cols]
        ex_dfs.append(ex_df)

    fig, axes = plt.subplots(1,3, figsize=(15,5))

    for i, ex_df in enumerate(ex_dfs):
        print('i', i)
        # pick all keypoints automatically
        keypoints = sorted({int(c.replace('xmean','')) for c in ex_df.columns if c.startswith('xmean')})

        # make a color map for each patient
        patients = ex_df['Patient_Id'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(patients)))
        color_map = dict(zip(patients, colors))

        for _, row in ex_df.iterrows():
            print('row', _)
            patient = row['Patient_Id']
            color = color_map[patient]

            for j in keypoints:
                x = row[f'xmean{j}']
                y = row[f'ymean{j}']
                xs = row[f'xstd{j}']
                ys = row[f'ystd{j}']

                # scatter point
                axes[i].scatter(x, y, color=color, label=patient, s=30, alpha=0.7, edgecolor='k')
                # label for point
                axes[i].text(x, y, j, color=color, fontsize=10, ha='right')
                # ellipse for std
                ellipse = Ellipse(
                    (x, y), width=2*xs, height=2*ys,
                    edgecolor=color, facecolor='none', lw=1.5, alpha=0.8
                )
                axes[i].add_patch(ellipse)

        # avoid duplicate legend entries
        handles, labels = axes[i].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axes[i].legend(unique.values(), unique.keys(), title="Patient_ID", bbox_to_anchor=(1.05, 1), loc='upper left')

        axes[i].set_xlabel("x mean")
        axes[i].set_ylabel("y mean")
        axes[i].set_title(f"Exercise {i}")
        axes[i].set_xlim(-0.5,0.5)
        axes[i].set_ylim(-1,1.5)
    plt.tight_layout()
    plt.show()
# weighted average patients
if 0:
    ex_dfs = []
    for i in range(3):
        ex_df = df_[df_['target']==i].drop(['target'],axis=1).iloc[:n,:][cols]
        #ex_df[keypoint_cols] = ex_df[keypoint_cols].mean(axis=0)
        ex_df['score'] = ex_score(ex_df, i)
        weights = ex_df['score']
        weighted_means = (ex_df[keypoint_cols].multiply(weights, axis=0).sum(axis=0)
                  / weights.sum())
        ex_df[keypoint_cols] = weighted_means
        ex_df = ex_df.iloc[:1,:]
        print(ex_df)
        ex_dfs.append(ex_df)

    #plots
    if 1:
        fig, axes = plt.subplots(1,3, figsize=(15,5))
        for i, ex_df in enumerate(ex_dfs):
            print('i', i)
            # pick all keypoints automatically
            keypoints = sorted({int(c.replace('xmean','')) for c in ex_df.columns if c.startswith('xmean')})

            for _, row in ex_df.iterrows():
                print('row', _)
                for j in keypoints:
                    x = row[f'xmean{j}']
                    y = row[f'ymean{j}']
                    xs = row[f'xstd{j}']
                    ys = row[f'ystd{j}']
                    print(xs, ys)

                    # scatter point
                    axes[i].scatter(x, y, s=30, alpha=0.7, edgecolor='k')
                    # label for point
                    axes[i].text(x, y, j, fontsize=10, ha='right')
                    # ellipse for std
                    ellipse = Ellipse(
                        (x, y), width=2*xs, height=2*ys, edgecolor='b',
                        facecolor='none', lw=1.5, alpha=0.8
                    )
                    axes[i].add_patch(ellipse)

            # avoid duplicate legend entries
            handles, labels = axes[i].get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            #axes[i].legend(unique.values(), unique.keys(), title="Patient_ID", bbox_to_anchor=(1.05, 1), loc='upper left')

            axes[i].set_xlabel("x mean")
            axes[i].set_ylabel("y mean")
            axes[i].set_title(f"Exercise {i}")
            axes[i].set_xlim(-0.5,0.5)
            axes[i].set_ylim(-1,1.5)
        plt.tight_layout()
        plt.show()
# Best exercise
if 1:
    k = 5
    ex_dfs = []
    for i in range(3):
        ex_df = df_[df_['target']==i].drop(['target'],axis=1).iloc[:n,:][cols]
        #ex_df[keypoint_cols] = ex_df[keypoint_cols].mean(axis=0)
        ex_df['score'] = ex_score(ex_df, i)
        ex_df = ex_df.sort_values(by='score', ascending=False).iloc[:k,:]
        print(ex_df.columns)
        #ex_df[keypoint_cols] = ex_df[keypoint_cols].mean(axis=0)
        #ex_df = ex_df.iloc[:1,:]
        print(ex_df)
        ex_dfs.append(ex_df)

    #plots
    if 1:
        fig, axes = plt.subplots(1,3, figsize=(15,5))
        for i, ex_df in enumerate(ex_dfs):
            print('i', i)
            # pick all keypoints automatically
            keypoints = sorted({int(c.replace('xmean','')) for c in ex_df.columns if c.startswith('xmean')})

            for _, row in ex_df.iterrows():
                print('row', _)
                for j in keypoints:
                    x = row[f'xmean{j}']
                    y = row[f'ymean{j}']
                    xs = row[f'xstd{j}']
                    ys = row[f'ystd{j}']
                    print(xs, ys)

                    # scatter point
                    axes[i].scatter(x, y, s=30, alpha=0.7, edgecolor='k')
                    # label for point
                    axes[i].text(x, y, j, fontsize=10, ha='right')
                    # ellipse for std
                    ellipse = Ellipse(
                        (x, y), width=2*xs, height=2*ys, edgecolor='b',
                        facecolor='none', lw=1.5, alpha=0.8
                    )
                    axes[i].add_patch(ellipse)

            # avoid duplicate legend entries
            handles, labels = axes[i].get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            #axes[i].legend(unique.values(), unique.keys(), title="Patient_ID", bbox_to_anchor=(1.05, 1), loc='upper left')

            axes[i].set_xlabel("x mean")
            axes[i].set_ylabel("y mean")
            axes[i].set_title(f"Exercise {i}")
            axes[i].set_xlim(-0.5,0.5)
            axes[i].set_ylim(-1,1.5)
        plt.tight_layout()
        plt.show()

'''
Classes are hard to distinguish,
If possible it would be good to give a score to how well each exercise is executed
so we can weight the samples for training
Some patients are left handed others are right handed ?
Scoring/Weights needs to be normalized by patient size
'''
# %% diffs
#hip 24 and 23
# shoulders 11 and 12
df_['hip_diff'] = np.abs(df_['xmean24'] - df_['xmean23'])
df_['shoulder_diff'] = np.abs(df_['xmean11'] - df_['xmean12'])
df_['torso_length'] = 0.5 *(np.abs(df_['ymean11'] - df_['ymean23']) + np.abs(df_['ymean12'] - df_['ymean24']))

diff_cols = ['hip_diff', 'shoulder_diff', 'torso_length']
def plot_f(ax, i):
    sns.histplot(df_[diff_cols[i]], ax=ax, kde=True)
    ax.set_title(diff_cols[i])
    ax.grid(True)
fig, axes = min_multiple_plot(len(diff_cols), plot_f)

hand_parts = {'lthumb':21, 'rthumb':22,'lindex':19,'rindex':20,'lpinky':17,'rpinky':18,'lwrist':15,'rwrist':16}
keypoint_handpart = {v:k for k,v in hand_parts.items()}
hand_diff_cols = []
for key, item in hand_parts.items():
    col = 'lmouth_'+key+'_diff'
    df_[col] = np.abs(df_['ymean9'] - df_['ymean'+str(item)])
    hand_diff_cols.append(col)

for key, item in hand_parts.items():
    col = 'rmouth_'+key+'_diff'
    df_[col] = np.abs(df_['ymean10'] - df_['ymean'+str(item)])
    hand_diff_cols.append(col)

def plot_f(ax, i):
    sns.violinplot(data=df_, ax=ax, x='target', y=hand_diff_cols[i])
    ax.set_title(hand_diff_cols[i])
    ax.grid(True)
fig, axes = min_multiple_plot(len(hand_diff_cols), plot_f)



def plot_f(ax, i):
    key = list(hand_parts.keys())[i]
    y_col = f'ystd{hand_parts[key]}'
    sns.violinplot(data=df_, ax=ax, x='target', y=y_col)
    ax.set_title(key + ' ystd')
    ax.grid(True)
fig, axes = min_multiple_plot(len(hand_parts.keys()), plot_f)

def plot_f(ax, i):
    key = list(hand_parts.keys())[i]
    y_col = f'xstd{hand_parts[key]}'
    sns.violinplot(data=df_, ax=ax, x='target', y=y_col)
    ax.set_title(key + ' xstd')
    ax.grid(True)
fig, axes = min_multiple_plot(len(hand_parts.keys()), plot_f)


plt.show()
# %% Impairment side

df_stroke = df_.copy()
for patient_id in range(1,15):
    for key, indexes in keypoint_side.items():
        cols = [txt + str(i) for txt in ['xstd', 'ystd'] for i in indexes]
        mask = df_stroke['Patient_Id']==patient_id
        df_stroke.loc[mask, key + 'std'] = df_stroke[mask][cols].sum().sum()
    

df_stroke['impairment_side'] = (df_stroke['leftstd'] > df_stroke['rightstd']).astype(int)
print(df_stroke[['Patient_Id','leftstd','rightstd','impairment_side']].drop_duplicates().sort_values(by='Patient_Id'))

df_long = df_stroke[['Patient_Id', 'leftstd', 'rightstd']].drop_duplicates()
df_long = (
    df_long
    .melt(id_vars='Patient_Id', 
          value_vars=['leftstd', 'rightstd'], 
          var_name='side', 
          value_name='std')
)

# Optional: clean up the 'side' column (remove the 'std' suffix)
df_long['side'] = df_long['side'].str.replace('std', '')
fig, axes = bar_plot(df_long, X='Patient_Id', y='std', label='side')
plt.show()
'''
Clear seperation of std values on left and right which most likely indicates impairment side!
'''
# %% face
#indexes = {'left':[1,2,3,7,9], 'right':[4,5,6,8,10]}
indexes = {'left':[], 'right':[], 'center':[0]}
left_face_cols = make_cols(indexes['left'])
right_face_cols = make_cols(indexes['right'])
center_face_cols = make_cols(indexes['center'])
face_df = df_[left_face_cols + right_face_cols + center_face_cols + ['Patient_Id', 'target']].copy()
#face_df = face_df[face_df['Patient_Id']==1]
xmean_cols = [col for col in face_df.columns if 'xmean' in col]
ymean_cols = [col for col in face_df.columns if 'ymean' in col]
for i in range(len(xmean_cols)):
    sns.scatterplot(data=face_df, x=xmean_cols[i], y=ymean_cols[i], hue='target')
plt.show()

# %% pacient visualization
if 1:
    torso = [11,12,24,23, 11]
    left_hand = [11, 13, 15, 17, 19, 15, 21]
    right_hand = [12, 14, 16, 18, 20, 16, 22]
    left_leg = [23,25,27,29,31,27]
    right_leg = [24,26,28,30,32,28]
    face = [7,3,2,1,0,4,5,6,8]
    mouth = [9,10]
    body_parts = [torso, left_hand, right_hand, left_leg, right_leg, face, mouth]
    body_side_parts = {'l': [23,11,13,15,17,19,21,9,7,1,2,3,0,25,27,29,31], 'r': [24,12,14,16,18,20,22,4,5,6,8,10,26,28,30,32]}
    
    #patients = [i for i in range(1,15)]
    patients = [3,6,1]
    #patients = [11,13,14]
    targets = [0,1]
    patient_ids = [id for _ in range(len(targets)) for id in patients ]
    targets_class = [t for t in targets for _ in range(len(patients))]
    def plot_patient(ax, j):
        cmap = plt.get_cmap('tab20').colors
        patient_id = patient_ids[j]
        target_class = targets_class[j]
        ax.set_title(f'Patient {patient_id} - Class {target_class}')
        
        if 1:
            X_sf = np.array(X['Skeleton_Features'].to_list())
            
            df_sf_ft = FeatureTransform3().fit_transform(X_sf)
            df_sf_ft[['Patient_Id','target']] = df_[['Patient_Id','target']].to_numpy()
            data = df_sf_ft.query(f'Patient_Id == {patient_id} & target == {target_class}')
 
        for body_part in body_parts:
            x_cols = [f'xmean{i}' for i in body_part]
            y_cols = [f'ymean{i}' for i in body_part]
            xpoints = data[x_cols].mean(axis=0).to_numpy()
            ypoints = data[y_cols].mean(axis=0).to_numpy()
            for std_index in [25,26,15,16]:
                xstd = data[f'xstd{std_index}'].mean(axis=0)
                ystd = data[f'ystd{std_index}'].mean(axis=0)
                xmean = data[f'xmean{std_index}'].mean(axis=0)
                ymean = data[f'ymean{std_index}'].mean(axis=0)
                ell = Ellipse(xy=(xmean, ymean), width=2*xstd, height=2*ystd,fill=False, linewidth=2)
                ax.add_patch(ell)
            ax.plot(xpoints, ypoints, marker='o', color=cmap[patient_id])

    if len(patients)>1:
        fig, axes = min_multiple_plot(len(patient_ids), plot_patient, n_cols=len(patients), n_rows=len(targets))
    else: fig,axes = min_multiple_plot(len(patient_ids), plot_patient)

    plt.show()



# %%
