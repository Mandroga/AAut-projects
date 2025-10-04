# %%
%run data_imports.py

# %% df info
print(df.iloc[0, 1])
print(df.info())
print(df.shape)
print(df['target'].unique())
print(df.iloc[0, 1].shape)
'''
df has 700 rows and 3 cols Patient id, skeleton features, target
target is categorical (exercise) values 0,1,2
each skeleton feature cell is an array (132,) 33 * 2 *2 (keypoints, (x,y), (mean,std))
'''
# %%
data = df.iloc[0, 1]
means = data[:66].reshape(2, 33).T
stds = data[66:].reshape(2, 33).T
data = pd.DataFrame({
    'x_mean': means[:, 0],
    'y_mean': means[:, 1],
    'x_std': stds[:, 0],
    'y_std': stds[:, 1],
    'keypoint': list(range(33))})

plt.plot(data['x_mean'], data['y_mean'], 'o')
plt.show()
plt.plot(data['x_std'], data['y_std'], 'o')
plt.show()



# %%
