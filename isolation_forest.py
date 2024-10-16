# Isolation Forest model
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import optuna
from sklearn.model_selection import train_test_split


# Load the dataset
df = pd.read_csv("/Users/anuradhadhawan/python_test/creditcard.csv")

# Define features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split the dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define StratifiedKFold on the training data
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 100, 300)
    max_samples = trial.suggest_float('max_samples', 0.4, 1.0)
    contamination = trial.suggest_float('contamination', 0.0016, 0.003)
    max_features = trial.suggest_float('max_features', 0.3, 1.0)

    # Initialize Isolation Forest with suggested hyperparameters
    model_if = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    f1_scores = []

    for train_index, valid_index in skf.split(X_train, y_train):
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        # Train Isolation Forest
        model_if.fit(X_train_fold)

        # Make predictions
        y_pred = model_if.predict(X_valid_fold)

        # Adjust the labels to match the original dataset's labels (0 = normal, 1 = fraud)
        y_pred_adjusted = np.where(y_pred == 1, 0, 1)

        # Calculate F1 score and collect it
        f1 = f1_score(y_valid_fold, y_pred_adjusted)
        f1_scores.append(f1)

    # Return the average F1 score across all folds
    return np.mean(f1_scores)

# Create Optuna study with SQLite storage
study = optuna.create_study(direction="maximize", storage="sqlite:///optuna_study.db")
study.optimize(objective, n_trials=32)

# Output the best hyperparameters
print(f"Best hyperparameters found: {study.best_params}")

# Initialize the best Isolation Forest model using the best hyperparameters
best_params = study.best_params
best_model_if = IsolationForest(
    n_estimators=best_params['n_estimators'],
    max_samples=best_params['max_samples'],
    contamination=best_params['contamination'],
    max_features=best_params['max_features'],
    random_state=42,
    n_jobs=-1
)

# Train on the full training set
best_model_if.fit(X_train)

# Make predictions on the test set
y_pred_test = best_model_if.predict(X_test)

# Adjust the labels
y_pred_test_adjusted = np.where(y_pred_test == 1, 0, 1)

# Calculate metrics on the test set
f1_test = f1_score(y_test, y_pred_test_adjusted)
recall_test = recall_score(y_test, y_pred_test_adjusted)

print(f"Test F1 Score: {f1_test}")
print(f"Test Recall:  {recall_test}")
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_adjusted))
print("\nTest Classification Report:\n", classification_report(y_test, y_pred_test_adjusted))
