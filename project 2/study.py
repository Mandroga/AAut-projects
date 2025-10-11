#%%
import pickle
import os

OUT_DIR = "optuna_results"
STUDY_PATH = os.path.join(OUT_DIR, "optuna_study.pkl")

# Load study
with open(STUDY_PATH, "rb") as f:
    study = pickle.load(f)

# Inspect best trial
print("Best trial number:", study.best_trial.number)
print("Best F1:", study.best_trial.value)
print("Best hyperparameters:", study.best_trial.params)

# View all trials in a DataFrame
df_trials = study.trials_dataframe()
print(df_trials.head())

# %%
