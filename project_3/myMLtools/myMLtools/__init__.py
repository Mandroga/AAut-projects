# aaut_tools/__init__.py
from .tools import *  # re-export all names from tools.py

# (Optional) a curated export list is cleaner:
# __all__ = [
#     "interactive_histogram_plotly",
#     "plot_covmatrix",
#     "plot_corrmatrix",
#     "plot_PCA",
#     "spread_analysis",
#     "plot_umap",
#     "plot_missingness",
#     "little_mcar",
#     "mar_test",
#     "missing_comb_plot",
#     "evaluate_imputer",
#     "dic_combinator",
#     "filter_df",
#     "min_multiple_plot_format",
#     "min_multiple_plot",
#     "bar_plot",
#     "get_line_by_label",
#     "line_by_label_filter",
#     "add_checkbox",
#     "plot_colors",
#     "plot_preds",
#     "plot_model_performance",
#     "pairs",
#     "nested_sgkf_bayes_report",
# ]
