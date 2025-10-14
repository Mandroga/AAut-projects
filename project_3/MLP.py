#%%
%run imports3.py

#%%
from classes import *
#%%
# --- Load data ---
with open("Xtrain2.pkl", "rb") as f:
    X = pickle.load(f)
y= np.load("Ytrain2.npy")
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
    # --- sample architecture ---
    n_layers = trial.suggest_int("n_layers", 1, 4)  # number of hidden layers

    n_neurons = []
    dropout = []
    for i in range(n_layers):
        n_neurons.append(trial.suggest_int(f"n_neurons_l{i}", 32, 512, step=32))
        dropout.append(trial.suggest_float(f"dropout_l{i}", 0.1, 0.6))
 
    # --- single activation for all hidden layers ---
    activation = trial.suggest_categorical("activation", ["swish", "relu", "tanh", "sigmoid"])

   # --- sample training hyperparameters ---
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 10, 60)
    threshold = trial.suggest_float("threshold", 0.01, 0.99)

    # --- class weights ---
    classes = np.unique(y)
    class_weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weight_dict = {c: w for c, w in zip(classes, class_weights)}

    # --- SciKeras wrapper ---
    clf = KerasClassifier(
        model=build_dynamic_mlp,
        model__input_dim=X.shape[1],
        model__num_classes=len(classes),
        model__n_layers=n_layers,
        model__n_neurons=n_neurons,
        model__dropout=dropout,
        model__activation=activation,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    pipe = Pipeline([("one hot", StringtoOneHotEncoder()),("scaler", StandardScaler()), ("nn", clf)])

    # --- CV evaluation ---
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_f1s = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pipe.fit(X_tr, y_tr, nn__class_weight=class_weight_dict)
        proba_val = pipe.predict_proba(X_val)[:, 1]
        y_pred = (proba_val >= threshold).astype(int)
        fold_f1s.append(f1_score(y_val, y_pred))

    mean_f1 = float(np.mean(fold_f1s))
    return mean_f1

# --- Running the study ---
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best trial params:")
for key, val in study.best_trial.params.items():
    print(f"{key}: {val}")
print("Best mean F1:", study.best_value)

# %%
