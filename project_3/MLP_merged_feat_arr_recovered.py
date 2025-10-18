#%%
%run imports3.py
#%run fitting3.py
#%%
from classes import *
import classes
#%%
# --- Load data ---
with open("Xtrain2.pkl", "rb") as f:
    X = pickle.load(f)
Y= np.load("Ytrain2.npy")

#%%
# --- Dynamic MLP builder for Optuna ---
def build_dynamic_mlp(input_dim, num_classes, n_layers=2, n_neurons=[128, 64], activation="swish", dropout=[0.3, 0.3]):
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for i in range(n_layers):
        x = layers.Dense(n_neurons[i], activation=activation)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout[i])(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[]
    )
    return model

# --- Optuna objective ---  architecture + single activation + threshold ---
def objective(trial):
    # --- Sample MLP architecture ---
    n_layers = trial.suggest_int("n_layers", 2, 3)
    n_neurons = []
    dropout = []
    for i in range(n_layers):
        n_neurons.append(trial.suggest_int(f"n_neurons_l{i}", 5, 12, step=1))
        dropout.append(trial.suggest_float(f"dropout_l{i}", 0.5, 0.6))

    activation = trial.suggest_categorical("activation", ["swish", "relu", "tanh", "sigmoid"])

    # --- Training hyperparameters ---
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 10, 60)
    threshold = trial.suggest_float("threshold", 0.01, 0.99)

    # --- Class weights ---
    classes = np.unique(Y)
    class_weights = compute_class_weight("balanced", classes=classes, y=Y)
    class_weight_dict = {c: w for c, w in zip(classes, class_weights)}

    # --- CV loop ---
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_baccs = []


    # --- Preprocessing pipeline for this fold ---
    preproc_pipe = Pipeline([
        #("feat", preprocess_data()),
        #("ohe", StringtoOneHotEncoder())
        ("feat", FeatEngMerge3())
    ])

    # Fit & transform data
    X_pp = preproc_pipe.fit_transform(X)
    #print(X_pp)
    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_pp, Y):
        X_tr_pp, X_val_pp = X_pp[train_idx], X_pp[val_idx]
        y_tr, y_val = Y[train_idx], Y[val_idx]

        # Compute input_dim after preprocessing
        input_dim = X_tr_pp.shape[1]

        # --- Build KerasClassifier with correct input_dim ---
        clf = KerasClassifier(
            model=build_dynamic_mlp,
            model__input_dim=input_dim,
            model__num_classes=len(classes),
            model__n_layers=n_layers,
            model__n_neurons=n_neurons,
            model__dropout=dropout,
            model__activation=activation,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # --- Full pipeline with scaler + classifier ---
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("nn", clf)
        ])

        # --- Fit the model ---
        pipe.fit(X_tr_pp, y_tr, nn__class_weight=class_weight_dict)

        # --- Predict probabilities on validation set ---
        proba_val = pipe.predict_proba(X_val_pp)[:, 1]
        y_pred = (proba_val >= threshold).astype(int)

        # --- Balanced accuracy per fold ---
        fold_baccs.append(balanced_accuracy_score(y_val, y_pred))

    # Return mean balanced accuracy across folds
    return float(np.mean(fold_baccs))


#%%
# --- Running the study ---
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best trial params:")
for key, val in study.best_trial.params.items():
    print(f"{key}: {val}")
print("Best mean BaAc:", study.best_value)

# %%
