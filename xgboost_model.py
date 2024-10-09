import optuna
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from optuna.trial import Trial
from optuna_dashboard import run_server

def load_data(path):
    data = pd.read_csv(path)

    x = data.drop(columns=['Class'])
    y = data['Class']

    return train_test_split(x, y, train_size=0.8, random_state=0)


def objective(trial: Trial, x_train, y_train):
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

    scores = cross_val_score(xgb_model, x_train, y_train, cv=5, scoring='f1')
    return scores.mean()


def optimized_confusion_matrix(x_train, y_train, x_test, y_test, params):
    best_xgb_model = XGBClassifier(**params)
    best_xgb_model.fit(x_train, y_train)

    print(confusion_matrix(y_test, best_xgb_model.predict(x_test)))


def main():
    x_train, x_test, y_train, y_test = load_data("creditcard.csv")
    n_study_trials = 100

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage, study_name='xgboost_tuning', direction='maximize')
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=n_study_trials)

    print(f"Best value: {study.best_value} (params: {study.best_params})")

    optimized_confusion_matrix(x_train, y_train, x_test, y_test, study.best_params)

    run_server(storage)

if __name__ == "__main__":
    main()