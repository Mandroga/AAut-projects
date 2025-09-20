# %% data and imports
%run imports_data.py
# %% data visualization and information

#info and describe
if 1:
    print(X.shape, y.shape)
    print(df.info())
    print(df.describe)

#data pairplot and Xi,Y scatter
if 1:
    def plot_f(subplot, n):
        cols = df_scaled.columns
        subplot.plot(df_scaled[cols[n]], df_scaled['target'], label=cols[n], marker='o', linestyle='', markersize=3, alpha=0.7)
        subplot.grid(True)
        subplot.legend()
    min_multiple_plot(6, plot_f)
    sns.pairplot(df,hue ='target', palette='viridis')
    plt.show()

print(counts := df['target'].abs().sort_values())

'''
From describe,
There's no missing values, nothing seems strange, mean close to median
From visualization, targets can be distinguished by some of the features with some noise

By visualizing the feature target scatter, there seems to be some relation with the target,
but the feature combination scatters seem to create a more clear distinction between targets!
try multilinear regression with combination of features - create polynomial features see if it becomes linear with target?

Some of the features seem highly correlated, try PCA and lasso for feature reduction, PLS regression

No zeros in target, but small values, 1e-4, smape ?
'''

# %% data histograms
data_for_hist = df_scaled


def plot_f(subplot, n):
    cols = data_for_hist.columns
    sns.histplot(data_for_hist[cols[n]], kde=True, ax=subplot)

min_multiple_plot(7, plot_f)
    
#sns.histplot(data_for_hist, kde=True)
plt.show()
interactive_histogram_plotly(data_for_hist, nbins=50)
plt.show()

'''
Histogram inspection:
0,2 - Seems unifor
1 - Seems normal, but theres a big peak which doesnt follow the distribution
3 - Curved around the edges, uniform in the middle (what is this distribution ??)
4,5 - Looks normal
target- seems skewed normal ?
'''


# %% Feature selection
'''
Three types of methods: Filter, embedded and wrapper 
We will use filter -  and wrapper
'''

#Filter methods - Corr with target, Var threshold, MI with target


#RFE feature selection
if 0:
    degrees = [1,2,4,6]
    linear_regressors = [LinearRegression()]
    for d in degrees:
        model = linear_regressors[0]
        
        poly = PolynomialFeatures(degree=d, include_bias=False)
        X_poly = pd.DataFrame(poly.fit_transform(X_df))
        feature_names = poly.get_feature_names_out(X_df.columns)
        X_poly.columns = feature_names

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        selector = RFECV(model, cv=cv, scoring='r2') 
        selector = selector.fit(X_poly, y)

        print("Optimal number of features:", selector.n_features_)
        print("Selected features:", list(X_poly.columns[selector.support_]))

'''

'''


 

# %%
