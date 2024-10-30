import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import optuna
from sklearn.model_selection import train_test_split
from optuna.trial import Trial

def get_train_test(path):
    df = pd.read_csv(path)

    X = df.drop('Class', axis=1)
    columns = ['V14','V17','V12', 'V10'] 
    X = df[columns]
    y = df['Class']

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def objective(trial: Trial, skf, x_train, y_train):
    n_estimators = trial.suggest_int('n_estimators', 100, 300)
    max_samples = trial.suggest_float('max_samples', 0.4, 1.0)
    contamination = trial.suggest_float('contamination', 0.0016, 0.002)
    max_features = trial.suggest_float('max_features', 0.3, 1.0)
    f1_scores = []

    model_if = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    train_model(f1_scores, model_if, skf, x_train, y_train)
    return np.mean(f1_scores)


def train_model(f1_scores, model: IsolationForest, skf, x_train, y_train):
    for train_index, valid_index in skf.split(x_train, y_train):
        x_train_fold, x_valid_fold = x_train.iloc[train_index], x_train.iloc[valid_index]
        _, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        model.fit(x_train_fold)

        y_pred = model.predict(x_valid_fold)

        y_pred_adjusted = np.where(y_pred == 1, 0, 1)

        f1 = f1_score(y_valid_fold, y_pred_adjusted)
        f1_scores.append(f1)


def test_best_model(model: IsolationForest, x_train, x_test, y_test):
    model.fit(x_train)

    y_pred_test = model.predict(x_test)
    y_pred_test_adjusted = np.where(y_pred_test == 1, 0, 1)

    print(f"Test F1 Score: {f1_score(y_test, y_pred_test_adjusted)}")
    print(f"Test Recall:  {recall_score(y_test, y_pred_test_adjusted)}")
    print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_adjusted))
    print("\nTest Classification Report:\n", classification_report(y_test, y_pred_test_adjusted))

def main():
    path = "creditcard.csv"

    x_train, x_test, y_train, y_test = get_train_test(path)
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    study = optuna.create_study(direction="maximize", storage="sqlite:///optuna_study.db")
    study.optimize(lambda trial: objective(trial, skf, x_train, y_train), n_trials=2)

    best_params = study.best_params
    print(f"Best hyperparameters found: {best_params}")

    test_best_model(
        IsolationForest(**best_params),
        x_train,
        x_test,
        y_test
    )


if __name__ == "__main__":
    main()