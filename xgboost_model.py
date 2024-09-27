import optuna
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from optuna.study.study import Trial

data = pd.read_csv("creditcard.csv")

x = data.drop(columns=['Class'])
y = data['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=2)

def objective(trial: Trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    gamma = trial.suggest_float('gamma', 0, 5)

    xgb_model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        random_state=0
    )

    xgb_model.fit(x_train, y_train)

    y_pred = xgb_model.predict(x_test)
    return f1_score(y_test, y_pred)

study = optuna.create_study(direction='maximize')  # Maximize the f1 score
study.optimize(objective, n_trials=100)

print(f"Best value: {study.best_value} (params: {study.best_params})")

best_xgb_model = XGBClassifier(**study.best_params)

best_xgb_model.fit(x_train, y_train)
y_pred = best_xgb_model.predict(x_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
