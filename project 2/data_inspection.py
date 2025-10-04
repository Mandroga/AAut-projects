# %%
%run data_imports.py

# %% df info
print(df.head(100))
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
data = df.iloc[328, 1]
means = data[:66]
x_means = means[:33]
y_means = means[33:]

x_means2 = means[::2]
y_means2 = means[1::2]

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].plot(x_means, y_means, 'o')
ax[1].plot(x_means2, y_means2, 'o')
plt.show()

# %%
