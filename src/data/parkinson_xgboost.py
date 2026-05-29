import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import  root_mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo

parkinsons_telemonitoring = fetch_ucirepo(id=189)

# seleção de alguns parâmetros
features = [
    "HNR",
    "PPE",
    "Jitter(%)",
    "Shimmer",
    "RPDE",
    "DFA"
]
X = parkinsons_telemonitoring.data.features
X = X[features]
y = parkinsons_telemonitoring.data.targets['total_UPDRS']

df = pd.concat([X, y], axis=1)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# train/test 0.7/0.3

def objective(trial):

    params = { # parâmetros para otimizçaão bayersiana
        "objective": "reg:squarederror",

        "n_estimators": trial.suggest_int(
            "n_estimators", 2000, 2000
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.03, log=True
        ),

        "max_depth": trial.suggest_int(
            "max_depth", 2, 5
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight", 5, 20
        ),

        "gamma": trial.suggest_float(
            "gamma", 1, 8
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda", 5, 50
        ),

        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
    }

    model = xgb.XGBRegressor(**params)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = root_mean_squared_error(
        y_test,
        preds,
    )

    return rmse

study = optuna.create_study(
    direction="minimize"
)

# otimização bayersiana
study.optimize(
    objective,
    n_trials=200
)

print(study.best_params)
print(study.best_value)

best_trial = study.best_trial

print(best_trial.value)
print(best_trial.params)


# pegar e executar novamente o melhor modelo
best_model = xgb.XGBRegressor(
    objective="reg:squarederror",

    **best_trial.params
)

best_model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_test, y_test)
    ],

    verbose=False
)



# Curvas de treino teste
results = best_model.evals_result()

train_rmse = results["validation_0"]["rmse"]
test_rmse = results["validation_1"]["rmse"]

plt.figure(figsize=(10,6))

plt.plot(train_rmse, label="train")
plt.plot(test_rmse, label="test")

plt.xlabel("Árvores")
plt.ylabel("RMSE")

plt.title("Melhor Trial - Curva RMSE")

plt.legend()

plt.show()



preds = best_model.predict(X_test)

plt.figure(figsize=(8,6))

plt.scatter(y_test, preds)

plt.xlabel("Real")
plt.ylabel("Predito")

plt.title("Melhor Trial - Real vs Predito")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()



residuals = y_test - preds

plt.figure(figsize=(8,6))

plt.scatter(preds, residuals)

plt.axhline(y=0)

plt.xlabel("Predição")
plt.ylabel("Resíduo")

plt.title("Melhor Trial - Resíduos")

plt.show()


# importância dos atributos
xgb.plot_importance(
    best_model,
    importance_type="gain",
    max_num_features=10
)

plt.show()

# Sortear melhores 10 trials (com menor RMSE)
best_trials = sorted(
    study.trials,
    key=lambda t: t.value
)[:10]
for i, trial in enumerate(best_trials):

    print(f"\nRank #{i+1}")

    print(f"RMSE: {trial.value}")

    print("Params:")

    for key, value in trial.params.items():
        print(f"  {key}: {value}")