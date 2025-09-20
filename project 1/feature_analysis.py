# %% data and imports
%run imports_data.py

'''
Three types of methods: Filter, embedded and wrapper 
We will use filter and wrapper

MI gives us information on non linear relations with target (?)
Corr gives us linear relation with target

High MI, low Corr -> different degree poly needed !?
We want features with high correlation !
'''

#Filter methods - Corr with target, Var threshold, MI with target
degree = 4

x_scaler = StandardScaler()
X_scaled = pd.DataFrame(x_scaler.fit_transform(X))
X_scaled.columns = X_scaled.columns.astype(str)

poly = PolynomialFeatures(degree=degree, include_bias=False)

X_poly = pd.DataFrame(poly.fit_transform(X_df))
feature_names = poly.get_feature_names_out(X_df.columns)
X_poly.columns = feature_names

X_scaled_poly = pd.DataFrame(poly.fit_transform(X_scaled))
feature_names = poly.get_feature_names_out(X_scaled.columns)
X_scaled_poly.columns = feature_names

poly_scaler = StandardScaler()
X_sps = pd.DataFrame(poly_scaler.fit_transform(X_scaled_poly), columns=X_scaled_poly.columns)

# %% PCA and CovMatrix original features
plot_PCA(df, ['target'])
plot_PCA(df.drop(['3'], axis=1), ['target'])
plot_PCA(df.drop(['3','5'], axis=1), ['target'])
plot_covmatrix(df)

#Testing Poly Features -- (is this useful?)
if 0:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_df = pd.DataFrame(X_train)
    X_train_df.columns = X_train_df.columns.astype(str)
    y_train_df = pd.DataFrame(y_train)
    degrees = [1, 2, 4, 6]
    all_corrs = []

    y_series = y_train_df.iloc[:, 0]  # target as Series

    for d in degrees:
        poly = PolynomialFeatures(degree=d, include_bias=False)
        X_poly = pd.DataFrame(poly.fit_transform(X_train_df))
        feature_names = poly.get_feature_names_out(X_train_df.columns)
        X_poly.columns = feature_names
        
        # Absolute correlations with target
        corrs = X_poly.apply(lambda x: x.corr(y_series)).values
        
        # Add degree info for plotting
        df_plot = pd.DataFrame({'Degree': [d]*len(corrs), 'Features':feature_names, 'Correlation': corrs})
        all_corrs.append(df_plot)

    # Combine all degrees
    plot_df = pd.concat(all_corrs, ignore_index=True)
    plot_df_abs = plot_df.copy()
    plot_df_abs = plot_df_abs.drop_duplicates(subset='Features', keep='first')
    plot_df_abs['Correlation'] = plot_df_abs['Correlation'].abs()

    plot_df_abs['feature_set'] = plot_df_abs['Features'].apply(lambda x: set(str(x).split()))

    # Keep only dominant feature sets
    dominant_rows = []

    for i, row_i in plot_df_abs.iterrows():
        is_dominated = False
        for j, row_j in plot_df_abs.iterrows():
            if i == j:
                continue
            # row_j set is a subset of row_i and has higher correlation
            if row_j['feature_set'].issubset(row_i['feature_set']) and row_j['Correlation'] > row_i['Correlation']:
                is_dominated = True
                break
        if not is_dominated:
            dominant_rows.append(i)

    plot_df_abs_dominant = plot_df_abs.loc[dominant_rows].reset_index(drop=True)
    pd.set_option('display.max_rows', None)      # None = no limit
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)    
    print(plot_df_abs_dominant.sort_values(by='Correlation', ascending=False))

     #Violin plot
    #sns.violinplot(x='Degree', y='Correlation', data=plot_df, inner="point")
    #plt.ylabel("Absolute Correlation with Target")
    #plt.show()
            
    #plot
    if 1:
        df = X_poly
        df['target'] = y_df

        def plot_f(subplot, n):
            cols = X_poly.columns
            subplot.plot(X_poly[cols[n]], X_poly['target'], label=cols[n], marker='o', linestyle='', markersize=3, alpha=0.7)
            subplot.grid(True)
            subplot.legend()
        #min_multiple_plot(len(X_poly.columns), plot_f)
        #sns.pairplot(df,hue ='target', palette='viridis')
        col_comb = list(itertools.combinations(df.drop('target',axis=1).columns, 2))
        print(col_comb)
        def plot_f2(subplot, n):
            c1,c2 = col_comb[n]
            sns.scatterplot(data=df,x=c1, y=c2,hue="target",palette="viridis",s=80,ax=subplot, legend=False)
            #subplot.set_title(f"x={c1}, y={c2}")
        min_multiple_plot(len(col_comb), plot_f2)
        plt.show()


'''
PCA plot shows that 2 components explain 0 variance, which matches our observation in the pairplot
where we see that two pairs of features are highly correlated

Beware! Features might not be correlated after Transformation function!

Visualizing Poly Features ---
2nd degree - squared features are not seperating very well, but some feature products sums are becoming linear with target
(x3*x4+x3*x5) and (x2*x4+x2*x5)

'''

# %% var thresh
# Var is linear, so for poly means we only need first features
cv = (X_df.std()/X_df.mean().abs())
print(cv.sort_values())
# smallest std is 10% of mean, no features dropped

# %% MI
# Data leakage from using whole dataset for feature selection ?!?!
mi = mutual_info_regression(X_df, y_df.iloc[:,0])
mi_series = pd.Series(mi, index=X_df.columns).sort_values(ascending=False)
print("Mutual Information (top features):")
print(mi_series)

#MI for feature combinations
def MI_poly_feature_comb(X_poly, feature):
    features_filtered = [col for col in X_poly.columns if feature in col and '^' not in col]
    X_poly_MIf = X_poly[features_filtered].copy().drop(feature, axis=1)
    features = X_poly_MIf.columns
    features_nofeat = [" ".join([n for n in feat.split(' ') if n != feature]) for feat in features]
    mi_poly = mutual_info_regression(X_poly_MIf, y_df.iloc[:,0])-mutual_info_regression(X_poly[features_nofeat], y_df.iloc[:,0])
    mi_poly_series = pd.Series(mi_poly, index=X_poly_MIf.columns).sort_values(ascending=False)
    return mi_poly_series

def plot_f(subplot, n):
    mi_poly_series = MI_poly_feature_comb(X_poly, str(n))
    sns.histplot(mi_poly_series, ax=subplot)
    subplot.set_title(n)

min_multiple_plot(6, plot_f)
plt.show()
'''
Feature 1 provides significantly less information comparing to others
Some feature combinations contribute to less MI others More
Features 0 and 3 have more counts on the positive side than others
meaning their combinations provide more information
the opposite happens for 2 and less intensely for 4 and 5
Analyze more!
'''
# %% Corr
corr_with_target = X_poly.corrwith(y_df.iloc[:,0]).abs()

print('#Features corr > threshold')
corr_target_threshold = [0.01, 0.05, 0.1, 1]
for ct in corr_target_threshold:
    n_features = (corr_with_target < ct).sum()
    print(f'corr < {ct}: {n_features}')
    
sns.histplot(corr_with_target.values, bins=np.arange(0,1.05,0.05))
plt.grid(True)
plt.show()
print(corr_with_target.sort_values().head())
'''
Very low corr features
Analyze!!!
'''
# %% correlation between features

corr_matrix = X_poly.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
print(corr_matrix.shape)

corr_threshold = [0.99, 0.95, 0.9, 0.5, 0]
print('#Features corr threshold')
for ct in corr_threshold:
    dropped = (upper.abs() > ct).sum().sum()
    print(f'{dropped} correlations > {ct}')


threshold = 0.5

# Create dictionary
corr_dict = {}
for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):  # only upper triangle
        if corr_matrix.iloc[i,j] < threshold:
            corr_dict[(corr_matrix.index[i], corr_matrix.columns[j])] = corr_matrix.iloc[i,j]

print(len(corr_dict))

'''

'''
# %% feature selection
# We will create different datasets, representative of transformed space (poly)
# filtered in different ways
