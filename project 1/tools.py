#imports

import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import cross_validate
from scipy.stats import chi2
from catboost import CatBoostClassifier, CatBoostRegressor

def interactive_histogram_plotly(df, x_label='X', y_label='Counts', title='', x_range=None, nbins=None, y_max=0):
    """
    Plota histogramas interativos com checkboxes dentro do gráfico.
    """

    # Limites automáticos do eixo X
    if x_range is None:
        x_min = df.min().min()
        x_max = df.max().max()
        x_range = [x_min, x_max]
    else:
        x_min, x_max = x_range

    # Define nbins se não fornecido
    if nbins is None:
        nbins = int(x_max - x_min) + 1

    # Cria edges de bins iguais
    bins = np.linspace(x_min, x_max, nbins + 1)

    fig = go.Figure()

    # Adiciona histogramas com bins fixos
    for col in df.columns:
        fig.add_trace(go.Histogram(
            x=df[col],
            name=col,
            opacity=0.75,
            visible=True,
            xbins=dict(
                start=x_min,
                end=x_max,
                size=(x_max - x_min) / nbins   # largura de cada bin
            )
        ))

    # Calcula y_max fixo com np.histogram usando os mesmos bins
    for col in df.columns:
        counts, _ = np.histogram(df[col].dropna(), bins=bins)
        y_max = max(y_max, counts.max())

    # margem de 10%
    y_max *= 1.1  

    # Cria botões
    buttons = []
    for i, col in enumerate(df.columns):
        visible = [False] * len(df.columns)
        visible[i] = True
        buttons.append(dict(
            label=col,
            method="update",
            args=[{"visible": visible}, {"title": f"Histograma: {col}"}]
        ))

    buttons.append(dict(
        label="Todas",
        method="update",
        args=[{"visible": [True]*len(df.columns)}, {"title": "Todos os Histogramas"}]
    ))

    # Layout
    fig.update_layout(
        updatemenus=[dict(
            active=len(df.columns),
            buttons=buttons,
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top'
        )],
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        barmode='overlay',
        xaxis=dict(range=x_range),
        yaxis=dict(range=[0, y_max])
    )

    #meter widget no sitio certo
    fig.update_layout( updatemenus=[dict(active=len(df.columns), buttons=buttons, x=1.1, y=0.0)], title=title, xaxis_title=x_label, yaxis_title=y_label, barmode='overlay', xaxis=dict(range=x_range), yaxis=dict(range=[0, y_max]) )

    fig.show()


def plot_covmatrix(df_):
    df_ = df_.select_dtypes(include='number')
    cols = df_.columns
    cov_mat = np.cov(df_.T)
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cov_mat,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 12},
                     yticklabels=cols,
                     xticklabels=cols)
    plt.title('Covariance matrix')
    plt.tight_layout()
    plt.show()

def plot_corrmatrix(df_):
    df_numerical = df_.select_dtypes(include='number')
    stsc = StandardScaler()
    cols = df_numerical.columns
    df = stsc.fit_transform(df_numerical[cols])
    cov_mat = np.cov(df.T)
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cov_mat,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 12},
                     yticklabels=cols,
                     xticklabels=cols)
    plt.title('Correlation matrix')
    plt.tight_layout()
    plt.show()

def plot_PCA(df, targets):

    X_numerical = df.select_dtypes(include='number').drop(targets,axis=1)
    stsc = StandardScaler()
    X = stsc.fit_transform(X_numerical)
    pca = PCA()
    pca.fit(X)

    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    # Plot the explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o',
             linestyle='--', label='explained variance per component', color='b')
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='r',label='cumulative explained variance')
    plt.title('Explained Variance per Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.legend()
    #plt.show()

def plot_covmatrix(df_):
    df_numerical = df_.select_dtypes(include='number')
    stsc = StandardScaler()
    cols = df_numerical.columns
    df = stsc.fit_transform(df_numerical[cols])
    cov_mat = np.cov(df.T)
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cov_mat,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 12},
                     yticklabels=cols,
                     xticklabels=cols)
    plt.title('Covariance matrix showing correlation coefficients')
    plt.tight_layout()
    plt.show()

def spread_analysis(df_, tail=0.05):
    """
    Returns std / (upper-quantile – lower-quantile) for each column.
    """
    df = df_.select_dtypes(include='number')
    lo = df.quantile(tail)
    hi = df.quantile(1-tail)
    prange = hi - lo
    return df.std(ddof=1) / prange

def plot_umap(df_, target, dimension='2d'):
    """
    Plots a UMAP projection of the given DataFrame with numerical target labels.

    Parameters:
    - df_: pandas DataFrame containing the features to be projected.
    - targets_: string or list of column names representing numerical target labels.
    - dimension: '2d' or '3d', specifies the dimensionality of the UMAP projection.
    """
    # Suppress FutureWarnings from sklearn
    #warnings.filterwarnings("ignore", category=FutureWarning)

    # Extract target values and features
    targets = df_[target].values.flatten()  # Ensure targets are 1D
    df = df_.select_dtypes(include='number')

    # Standardize the feature data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # UMAP and Plotting
    if dimension == '2d':
        umap_model = umap.UMAP(n_components=2)
        umap_result = umap_model.fit_transform(df_scaled)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=targets, cmap='viridis', edgecolor='k', alpha=0.7)
        plt.title('2D UMAP Projection with Targets')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.colorbar(scatter, label='Target Value')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif dimension == '3d':
        from mpl_toolkits.mplot3d import Axes3D  # Only needed for 3D plot
        umap_model = umap.UMAP(n_components=3)
        umap_result = umap_model.fit_transform(df_scaled)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], c=targets, cmap='viridis',
                             edgecolor='k', alpha=0.7)
        ax.set_title('3D UMAP Projection with Targets')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        fig.colorbar(scatter, label='Target Value')
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("Dimension must be '2d' or '3d'")

def plot_PCA(df, targets=None):
    if targets == None:
        X_numerical = df.select_dtypes(include='number')
    else:
        X_numerical = df.select_dtypes(include='number').drop(targets,axis=1)
    stsc = StandardScaler()
    X = stsc.fit_transform(X_numerical)
    pca = PCA()
    pca.fit(X)

    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    # Plot the explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o',
             linestyle='--', label='explained variance per component', color='b')
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='r',label='cumulative explained variance')
    plt.title('Explained Variance per Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_missingness(df):
    """
    Plots a binary heatmap of missing values in the DataFrame.
    Missing cells are shown in one color, present cells in another.
    """
    # Create the boolean mask: True for missing, False for present
    mask = df.isna().values

    plt.figure(figsize=(12, 6))
    # imshow will plot True as 1, False as 0
    plt.imshow(mask, aspect='auto', interpolation='none')

    # Label axes
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    # Show column names on x-axis
    plt.xticks(
        ticks=range(len(df.columns)),
        labels=df.columns,
        rotation=90
    )

    # Add a colorbar with custom tick labels
    cbar = plt.colorbar()
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Present', 'Missing'])

    plt.title('Missing Data Visualization')
    plt.tight_layout()

#missing data
def little_mcar(df: pd.DataFrame, target: str = None):
    """
    Run a Little’s‐style MCAR test *per column*.

    For each column C with any NaNs:
      • Let y = 1{C is missing}, 0{C is observed}.
      • Let X = all other columns.
      • Drop any rows where X has NaNs (so we can compute covariances cleanly).
      • Compute the Mahalanobis‐type statistic S = n_missing * (μ_miss - μ_obs)^T Σ⁻¹ (μ_miss - μ_obs)
      • df = number of columns in X
      • p_value = 1 - CDF_χ²(S; df)
      • If p_value ≥ 0.05 → fail to reject MCAR for column C; else → not MCAR.

    Returns
    -------
    Dict[col, Dict[str, float or bool]]
      {
        'chi-square': S,
        'df': df,
        'p-value': p_value,
        'MCAR?': (p_value >= 0.05)
      }
    """
    df = df.copy()
    if target:
        df.drop(columns=[target], inplace=True, errors='ignore')

    results = {}
    # only iterate over columns that actually have missing values
    for col in df.columns[df.isna().any()]:
        y = df[col].isna().astype(int)
        X = df.drop(columns=[col])

        # drop rows where any predictor is missing
        idx = X.dropna().index
        X = X.loc[idx]
        y = y.loc[idx]

        # if no missing or no non‐missing left, or too few predictors → skip
        n_miss = int(y.sum())
        p = X.shape[1]
        if n_miss == 0 or n_miss == len(y) or p < 1:
            results[col] = {
                'chi-square': np.nan,
                'df': np.nan,
                'p-value': np.nan,
                'MCAR?': None
            }
            continue

        # group means
        mu_obs  = X.loc[y == 0].mean()
        mu_miss = X.loc[y == 1].mean()
        cov     = X.cov()

        # difference in means
        diff = (mu_miss - mu_obs).values.reshape(-1, 1)  # shape (p, 1)

        # pseudo‐inverse in case cov is singular
        inv_cov = np.linalg.pinv(cov.values)

        # test statistic
        S = float(n_miss * (diff.T @ inv_cov @ diff))
        df_stat = p
        p_value = chi2.sf(S, df_stat)

        results[col] = {
            'chi-square': S,
            'df': df_stat,
            'p-value': p_value,
            'MCAR?': p_value >= 0.05
        }
    print(pd.DataFrame(results))
    return results

def mar_test(df_, target=None, scoring_metrics=['roc_auc']):
    print('')
    df = df_.copy()
    if target:
        df = df.drop([target], axis=1)
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['category']).columns.tolist()
    missing_cols = df.columns[df.isna().any()].tolist()
    score_df = pd.DataFrame(columns=['col','scoring_metric','set','mean score', 'std score'])
    for missing_col in missing_cols:
        y = df[missing_col].isnull().astype(int)
        X = df.drop([missing_col], axis=1)
        model = CatBoostClassifier(verbose=0)
        cross_val_score = cross_validate(model, X, y, cv=2, scoring=scoring_metrics, return_train_score=True)
        for dataset in ['train','test']:
            for scoring_metric in scoring_metrics:
                score_df.loc[len(score_df)] = [missing_col, scoring_metric, dataset, np.mean(cross_val_score[f'{dataset}_{scoring_metric}']), np.std(cross_val_score[f'{dataset}_{scoring_metric}'],ddof=1)]
    bar_plot(score_df, min_multiples='scoring_metric', X='set',y='mean score',label='col')
    plt.show()
    return score_df

def missing_comb_plot(df_):
    df = df_.copy()
    bool_df = df.isna().astype('int')
    bool_df['comb'] = bool_df.values.tolist()
    bool_df['comb'] = bool_df['comb'].astype(str).astype('category')
    category_counts = bool_df['comb'].value_counts()
    category_counts.plot(kind='bar', edgecolor='black')
    plt.grid(True)
    plt.tick_params(axis='x', rotation=90)
    plt.xticks(fontsize=7)
    plt.tight_layout()
    plt.title('Missing combination hist')
    plt.show()



def evaluate_imputer(df: pd.DataFrame,imputer = CatBoostRegressor(),sample_frac: float = 0.2,n_repeats: int = 5,random_state: int = None) -> pd.Series:
    """
    Evaluate an imputer by randomly masking known values and scoring per-column.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame with missing values.
    imputer : object
        Fitted or unfitted imputer with .fit_transform() method.
    sample_frac : float, default=0.2
        Fraction of non-missing entries in each column to mask per repeat.
    n_repeats : int, default=5
        Number of masking/imputation repeats to average the score.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    pd.Series
        Average per-column score (RMSE for numeric, accuracy for categorical).
    """
    rng = np.random.default_rng(random_state)
    cols = df.columns
    all_scores = {col: [] for col in cols}

    for _ in range(n_repeats):
        # 1) Mask a random subset of known values
        df_missing = df.copy()
        mask_indices = {}
        for col in cols:
            non_na_idxs = df.index[df[col].notna()]
            n_mask = int(np.floor(len(non_na_idxs) * sample_frac))
            if n_mask > 0:
                chosen = rng.choice(non_na_idxs, size=n_mask, replace=False)
                df_missing.loc[chosen, col] = np.nan
                mask_indices[col] = chosen
            else:
                mask_indices[col] = []

        # 2) Impute the masked DataFrame
        imputed_arr = imputer.fit(df_missing)
        imputed_df = pd.DataFrame(imputed_arr, columns=cols, index=df.index)

        # 3) Score per column
        for col in cols:
            idxs = mask_indices[col]
            if len(idxs) == 0:
                continue

            true_vals = df.loc[idxs, col]
            imputed_vals = imputed_df.loc[idxs, col]

            if pd.api.types.is_numeric_dtype(df[col]):
                score = mean_squared_error(true_vals, imputed_vals, squared=False)
            else:
                score = accuracy_score(true_vals.astype(str), imputed_vals.astype(str))

            all_scores[col].append(score)

    # 4) Compute average scores
    avg_scores = {
        col: (np.nan if not scores else np.mean(scores))
        for col, scores in all_scores.items()
    }
    return pd.Series(avg_scores, name="avg_score")
#Functional
def dic_combinator(dic_combinations):
    combinations = []
    if 1:
        keys = [k for k in dic_combinations]
        comb_lens = [len(dic_combinations[k]) for k in keys]
        comb_iter = [0] * len(comb_lens)
        i = 0

        def search_combinations(comb_iter, comb_lens):
            for i in range(len(comb_iter)):
                if comb_iter[i] == comb_lens[i]:
                    if (i + 1) == len(comb_lens):
                        return 2, comb_iter
                    else:
                        comb_iter[i + 1] += 1
                        for j in range(i + 1):
                            comb_iter[j] = 0
                        return 1, comb_iter
            return 0, comb_iter

        while True:
            result, comb_iter = search_combinations(comb_iter, comb_lens)
            if result == 0:
                combinations += [{keys[i]: dic_combinations[keys[i]][comb_iter[i]] for i in range(len(keys))}]
                comb_iter[0] += 1
            elif result == 2:
                return combinations

def filter_df(df_, IQR_multiplier_i=3):
    df = df_.copy()
    # IQR filter
    if 1:
        IQR_multipliers = np.array([1.5] + list(range(2, 10)) + [1e1, 1e2, 1e3, 1e4])
        outliers = []
        quantile = 0.25
        Q1 = df.quantile(quantile)
        Q3 = df.quantile(1 - quantile)
        # Calculate IQR (Interquartile Range)
        IQR = Q3 - Q1
        # Define outlier threshold (1.5 * IQR is common)
        for IQR_multiplier in IQR_multipliers:
            lower_bound = Q1 - IQR_multiplier * IQR
            upper_bound = Q3 + IQR_multiplier * IQR

            # Create a mask for values within the IQR
            mask = (df >= lower_bound) & (df <= upper_bound)

            # Filter out rows where any column has an outlier
            outlier_df = df[~mask.all(axis=1)]
            outliers += [len(outlier_df)]

        outliers = np.array(outliers)
        percentage_outliers = outliers / len(df)
        multipliers_outliers = {'IQR_multipliers': IQR_multipliers, 'number of outliers': outliers}
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)  # Prevent line wrapping
        pd.set_option('display.max_colwidth', None)
        print(pd.DataFrame(multipliers_outliers).T.to_string(header=False))
        # IQR_multiplier = np.min(np.array(IQR_multipliers)[percentage_outliers <= 0.1])
        IQR_multiplier = IQR_multiplier_i
        lower_bound = Q1 - IQR_multiplier * IQR
        upper_bound = Q3 + IQR_multiplier * IQR
        mask = (df >= lower_bound) & (df <= upper_bound)
    filtered_df = df_[mask.all(axis=1)]
    outlier_df = df_[~mask.all(axis=1)]
    return filtered_df, outlier_df

#Plots
def min_multiple_plot_format(N_plots, n_rows=None, n_cols=None):
    if n_rows == None and n_cols == None:
        plot_rows = int(np.floor(np.sqrt(N_plots)))
        col_ratio = N_plots / plot_rows
        plot_cols = int(np.floor(col_ratio))
        xtra_cols = int((col_ratio - plot_cols) * plot_rows)

        subplot_format = [plot_cols] * plot_rows
        subplot_format[:xtra_cols] = [x + 1 for x in subplot_format[:xtra_cols]]
       
    elif n_cols == None:
        n_cols = int(np.ceil(N_plots/n_rows))
        subplot_format = [n_cols]*n_rows
    elif n_rows == None:
         n_rows = int(np.ceil(N_plots/n_cols))
         subplot_format = [n_cols]*n_rows
    else:
        subplot_format = [n_cols]*n_rows
    return subplot_format
def min_multiple_plot(N_plots, plot_functions, check_box=False, n_rows=None, n_cols=None):
    plots = []
    subplot_format = min_multiple_plot_format(N_plots, n_rows=n_rows, n_cols=n_cols)
    fig, axes = plt.subplots(len(subplot_format), max(subplot_format))
    for i in range(len(subplot_format)):
        for j in range(subplot_format[i]):
            n = sum(subplot_format[:i]) + j
            if len(subplot_format) == 1:
                if subplot_format[0] == 1: subplot = axes
                else: subplot = axes[j]
            else:
                subplot = axes[i, j]

            plot_functions(subplot, n)
            plots += [subplot]

    
    def on_click(event):
        axes_flatten = list(axes.flatten())
        if event.inaxes not in axes_flatten:
            return
        ax_index = axes_flatten.index(event.inaxes)
        fig2, ax2 = plt.subplots()
        plot_functions(ax2, ax_index)
        plt.tight_layout()
        plt.show()


    fig.canvas.mpl_connect('button_press_event', on_click)

    if check_box:
        
        line_by_label = {}
        for subplot in plots:
            for line in subplot.get_lines():
                label = line.get_label()
                if line_by_label.get(label, None) == None: line_by_label[label] = []
                line_by_label[label].append(line)
            for scatter in subplot.collections:
                label = scatter.get_label()
                if line_by_label.get(label, None) == None: line_by_label[label] = []
                line_by_label[label].append(scatter)
        labels = [k for k in line_by_label]
        print(labels)
        if labels != []:
            rax = plt.axes([0.91, 0.05, 0.08, 0.1])
            visibility = [True] * len(labels)
            check = matplotlib.widgets.CheckButtons(rax, labels, visibility)
            def toggle_visibility(label):
                for plot in line_by_label[label]:
                    plot.set_visible(not plot.get_visible())
                plt.draw()

            check.on_clicked(toggle_visibility)
        plt.show()
    return fig, axes

def bar_plot(df_, y, X=None, label=None, min_multiples=None ,cmap_name='viridis', n_rows=None, n_cols=None):
    df = df_.copy()
    cat_cols = [X, label, min_multiples]
    cat_cols = [c for c in cat_cols if c!=None]
    df[cat_cols] = df[cat_cols].fillna(value='None').astype('category')

    import matplotlib.patches as mpatches

    def plot_f_xlmm(subplot, n):
        print(n)
        n_bars = len(label_unique)
        n_groups = len(X_unique)
        bar_width = 0.1
        space_width = 0.5
        ticks = []
        bar_pos = []
        heights = []
        bar_colors = []
        cm = plt.colormaps.get_cmap(cmap_name)

        # Assign one color per label
        color_map = {label_i: cm(i / len(label_unique)) for i, label_i in enumerate(label_unique)}

        init_pos = 0
        for xi in X_unique:
            mmi = min_multiples_unique[n]
            dfi = df[(df[X] == xi) & (df[min_multiples] == mmi)]

            # Center tick in the middle of group
            ticks.append(init_pos + (n_bars * bar_width) / 2)

            for j, label_i in enumerate(label_unique):
                value = dfi[dfi[label] == label_i][y].values
                print(value, xi, mmi, label_i)
                if len(value) > 1:
                    raise ValueError('Incorrect slicing: more than one value found.')
                elif len(value) == 0:
                    value = [0]  # or np.nan if you want to skip them

                heights.append(value[0])
                bar_pos.append(init_pos + j * bar_width)
                bar_colors.append(color_map[label_i])

            init_pos = bar_pos[-1] + space_width

        # Plot all bars
        bars = subplot.bar(
            x=bar_pos,
            height=heights,
            width=bar_width,
            color=bar_colors
        )

        #Bar numbers
        bar_numbers(subplot, bars)

        # Create one legend entry per unique label
        handles = [mpatches.Patch(color=color_map[lbl], label=lbl) for lbl in label_unique]
        subplot.legend(
            handles=handles,
            loc='lower left',  # position relative to the axes
            bbox_to_anchor=(0.01, 0.01),  # fine-tune exact placement (x=0.01, y=0.01 from bottom-left)
            frameon=True,  # show legend box
            fontsize='small'  # optional: smaller font
        )

        subplot.set_title(min_multiples_unique[n])
        subplot.set_xticks(ticks)
        subplot.set_xticklabels(X_unique)
        subplot.set_xlabel(X)
        subplot.grid(True)

    def plot_f_lmm(subplot, n):
        n_bars = len(label_unique)
        bar_width = 0.1
        space_width = 0.5
        ticks = []
        bar_pos = []
        heights = []
        bar_colors = []
        cm = plt.colormaps.get_cmap(cmap_name)

        # Assign one color per label
        color_map = {label_i: cm(i / len(label_unique)) for i, label_i in enumerate(label_unique)}

        init_pos = 0

        mmi = min_multiples_unique[n]
        dfi = df[(df[min_multiples] == mmi)]

        # Center tick in the middle of group
        ticks.append(init_pos + (n_bars * bar_width) / 2)

        for j, label_i in enumerate(label_unique):
            slice_df = dfi[dfi[label] == label_i][y]
            value = slice_df.values
            if len(value) > 1:
                raise ValueError(f'Incorrect slicing: more than one value found - {slice_df}')
            elif len(value) == 0:
                value = [0]  # or np.nan if you want to skip them

            heights.append(value[0])
            bar_pos.append(init_pos + j * bar_width)
            bar_colors.append(color_map[label_i])

        # Plot all bars
        bars = subplot.bar(
            x=bar_pos,
            height=heights,
            width=bar_width,
            color=bar_colors,
           # labels=label_unique
        )
        handles = [mpatches.Patch(color=color_map[lbl], label=lbl) for lbl in label_unique]
        subplot.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.01, 0.01), fontsize='small')

        #Bar numbers
        bar_numbers(subplot, bars)

        subplot.set_title(min_multiples_unique[n])
        # subplot.set_xticks(ticks)
        subplot.grid(True)

    def plot_f_xl(subplot, n):
        n_bars = len(label_unique)
        n_groups = len(X_unique)
        bar_width = 0.1
        space_width = 0.5
        ticks = []
        bar_pos = []
        heights = []
        bar_colors = []
        cm = plt.colormaps.get_cmap(cmap_name)

        # Assign one color per label
        color_map = {label_i: cm(i / len(label_unique)) for i, label_i in enumerate(label_unique)}

        init_pos = 0
        for xi in X_unique:
            dfi = df[df[X] == xi]

            # Center tick in the middle of group
            ticks.append(init_pos + (n_bars * bar_width) / 2)

            for j, label_i in enumerate(label_unique):
                value = dfi[dfi[label] == label_i][y].values
                if len(value) > 1:
                    raise ValueError(f"Multiple values found for X={xi}, label={label_i}")
                elif len(value) == 0:
                    value = [0]

                heights.append(value[0])
                bar_pos.append(init_pos + j * bar_width)
                bar_colors.append(color_map[label_i])

            init_pos = bar_pos[-1] + space_width

        # Plot all bars
        bars = subplot.bar(
            x=bar_pos,
            height=heights,
            width=bar_width,
            color=bar_colors
        )

        #Bar numbers
        bar_numbers(subplot, bars)
        # Legend
        handles = [mpatches.Patch(color=color_map[lbl], label=lbl) for lbl in label_unique]
        subplot.legend(
            handles=handles,
            loc='lower left',
            bbox_to_anchor=(0.01, 0.01),
            frameon=True,
            fontsize='small'
        )

        subplot.set_xticks(ticks)
        subplot.set_xticklabels(X_unique)
        subplot.set_xlabel(X)
        subplot.set_title("Bar Plot")  # pode personalizar
        subplot.grid(True)

    def bar_numbers(subplot, bars):
        # Add values above bars
        for bar in bars:
            height = bar.get_height()
            subplot.text(
                bar.get_x() + bar.get_width() / 2,  # x position (center of bar)
                height,                              # y position (top of bar)
                f'{height:.2f}',                     # label text
                ha='center',                         # horizontal alignment
                va='bottom',                         # vertical alignment
                fontsize=8                            # optional font size
            )

    if 1:
        if X: X_unique = df[X].unique()
        if label: label_unique = df[label].unique()
        if min_multiples: min_multiples_unique = df[min_multiples].unique()
        if X and label and min_multiples:
            N = len(min_multiples_unique)
            plot_f = plot_f_xlmm
        elif X and label:
            plot_f = plot_f_xl
            N = 1
        elif label and min_multiples:
            plot_f = plot_f_lmm
            N = len(min_multiples_unique)

    fig, axes = min_multiple_plot(N, plot_f, n_rows=n_rows, n_cols=n_cols)

    return fig, axes


# Checkboxes get plotted objects (lines and scatters) to labels
def get_line_by_label(axes):
    line_by_label = {}
    for subplot in axes.flatten():
        for line in subplot.get_lines():
            label = line.get_label()
            if line_by_label.get(label, None) == None: line_by_label[label] = []
            line_by_label[label].append(line)
        for scatter in subplot.collections:
            label = scatter.get_label()
            if line_by_label.get(label, None) == None: line_by_label[label] = []
            line_by_label[label].append(scatter)
    return line_by_label

def line_by_label_filter(line_by_label_, str_filter=['']):
    '''
    If subplots have been labeled and have a same string, you can group them,
    for example Train and test sets str_filter = ['Train','Test']
    '''
    line_by_label = {x: [item for key, line in line_by_label_.items() if x in key for item in (line if isinstance(line, list) else [line])]for x in str_filter}
    return line_by_label

    def on_draw(event):
        labels = list(line_by_label.keys())
        rax = fig.add_axes([0.85, 0.05, 0.12, 0.2])  # checkbox axes
        visibility = [True] * len(labels)
        check = matplotlib.widgets.CheckButtons(rax, labels, visibility)

        def toggle_visibility(label):
            for plot in line_by_label[label]:
                plot.set_visible(not plot.get_visible())
            fig.canvas.draw()  # redraw figure

        check.on_clicked(toggle_visibility)
        
        fig.canvas.draw()  # draw checkboxes initially
        fig.canvas.mpl_disconnect(cid)  # disconnect after first draw

    # Connect the draw_event
    cid = fig.canvas.mpl_connect('draw_event', on_draw)


def add_checkbox(line_by_label):
    labels = [k for k in line_by_label]
    rax = plt.axes([0.91, 0.05, 0.08, 0.1])
    visibility = [True] * len(labels)
    check = matplotlib.widgets.CheckButtons(rax, labels, visibility)
    def toggle_visibility(label):
        for plot in line_by_label[label]:
            plot.set_visible(not plot.get_visible())
        plt.draw()
    check.on_clicked(toggle_visibility)
    plt.show()
#-----------

def plot_colors(n_groups, n_elements=1, cmap_name='viridis'):
    cmap = plt.colormaps.get_cmap(cmap_name)
    return [cmap(i/n_groups) for i in range(n_groups) for _ in range(n_elements)]


def plot_preds(model_fit_data):
    regressor_names = [k for k in model_fit_data]
    def plot_funcs(subplot, n):
        name = regressor_names[n]
        data = model_fit_data[name]
        subplot.scatter(data['train'], data['train_pred'], s=7, label='train')
        subplot.scatter(data['test'], data['test_pred'], s=7, label='test')
        subplot.plot([min(data['train']), max(data['train'])], [min(data['train']), max(data['train'])], color='r',
                     linestyle='--', label="Perfect Prediction")
        subplot.legend()
        subplot.grid(True)
        subplot.set_title(name)

    min_multiple_plot(len(regressor_names), plot_funcs, True)

def plot_model_performance(model_fit_data, metrics):
    """
    model_fit_data: dict with keys 'train', 'train_preds', 'test', 'test_preds'
    metrics: list of strings: ['accuracy', 'precision', 'recall', 'f1']
    """
    y_train = model_fit_data['train']
    y_train_preds = model_fit_data['train_preds']
    y_test = model_fit_data['test']
    y_test_preds = model_fit_data['test_preds']

    # Mapping of metric names to functions
    metric_funcs = {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
    }

    # Compute scores
    scores = {'train': {}, 'test': {}}
    for metric in metrics:
        scorer = metric_funcs.get(metric)
        if scorer:
            scores['train'][metric] = scorer(y_train, y_train_preds)
            scores['test'][metric] = scorer(y_test, y_test_preds)

    # Plot Confusion Matrices
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y_true, y_pred, title in zip(
            axs,
            [y_train, y_test],
            [y_train_preds, y_test_preds],
            ['Train', 'Test']
    ):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{title} Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Plot Metric Barplots
    labels = list(scores['train'].keys())
    train_scores = [scores['train'][m] for m in labels]
    test_scores = [scores['test'][m] for m in labels]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x, train_scores, width, label='Train')
    plt.bar([p + width for p in x], test_scores, width, label='Test')
    plt.xticks([p + width / 2 for p in x], labels)
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.title('Model Performance')
    plt.legend()
    plt.tight_layout()
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def pairs(df, target=None):
    """
    R-style pairs plot with histograms, scatterplots, correlations,
    optional coloring (target).
    """

    # correlation annotation function
    def corrfunc(x, y, **kws):
        r = np.corrcoef(x, y)[0, 1]
        ax = plt.gca()
        ax.annotate(f"r = {r:.2f}", xy=(0.5, 0.5), xycoords=ax.transAxes,
                    ha='center', va='center', fontsize=12)

    # if target is given, treat it as hue
    hue = target if target in df.columns else None

    g = sns.PairGrid(df, hue=hue)

    # lower triangle → scatterplots
    g.map_lower(sns.scatterplot)

    # diagonal → histograms
    g.map_diag(sns.histplot)

    # upper triangle → correlations
    g.map_upper(corrfunc)

    if hue is not None:
        g.add_legend()

    plt.show()
    """
    R-style pairs plot with histograms, scatterplots, correlations,
    optional coloring (hue), and optional point labels (target).
    """
    
    def corrfunc(x, y, **kws):
        r = np.corrcoef(x, y)[0, 1]
        ax = plt.gca()
        ax.annotate(f"r = {r:.2f}", xy=(0.5, 0.5), xycoords=ax.transAxes,
                    ha='center', va='center', fontsize=12)
    
    g = sns.PairGrid(df, hue=hue)
    g.map_lower(sns.scatterplot)
    
    # If target labels are provided, annotate points
    if target is not None:
        def label_points(x, y, **kws):
            ax = plt.gca()
            ax.scatter(x, y, **kws)  # plot points
            for i, txt in enumerate(target):
                ax.text(x[i], y[i], str(txt), fontsize=8)
        g.map_lower(label_points)
    
    g.map_diag(sns.histplot)
    g.map_upper(corrfunc)
    
    if hue is not None:
        g.add_legend()

