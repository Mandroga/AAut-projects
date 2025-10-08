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
def plot_f(subplot, i):
    sns.histplot(df_[diff_cols[i]], ax=subplot)
    ax.set_title(diff_cols[i])
    ax.grid(True)
    
fig, axes = min_multiple_plot(len(diff_cols), plot_f)
plt.show()
# %% Impairment side
keypoint_side = {'left':[4,5,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
                 'right':[1,2,3,7,9,11,13,15,17,19,21,23,25,27,29,31]
                }
for key, indexes in keypoint_side.items():
    left_mean_std = df_[[txt+str(i) for txt in ['xstd','ystd'] for i in indexes]].groupby('Patient_Id').mean()
    print(left_mean_std)

# %%
