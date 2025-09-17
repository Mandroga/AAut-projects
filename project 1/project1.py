#import libraries
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import copy
from tools import *
# %%
#import data

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

X_train_df = pd.DataFrame(X_train)
y_train_df = pd.DataFrame(y_train)

df = X_train_df.copy()
df['target'] = y_train_df


x_scaler = StandardScaler()
y_scaler = StandardScaler()

df_scaled = df.copy()
feature_names = [col for col in df.columns if col != 'target']
df_scaled[feature_names] = x_scaler.fit_transform(df_scaled[feature_names])
df_scaled['target'] = y_scaler.fit_transform(df_scaled[['target']])
# %%
#Data inspection

print(X_train.shape, y_train.shape)
print(X_train_df.info())
print(y_train_df.info())
print(X_train_df.describe())
print(y_train_df.describe())

sns.pairplot(df,hue ='target', palette='viridis')
#pairs(df_scaled,target ='target', hue='viridis')
plt.show()

# %%
#sns.histplot(df, kde=True)
#plt.show()
#sns.histplot(df_scaled, kde=True)
print(df_scaled.min().min(),df_scaled[feature_names].max().max())

interactive_histogram_plotly(df_scaled, nbins=50)
plt.show()

# %%
