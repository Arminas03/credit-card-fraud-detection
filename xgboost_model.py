from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("creditcard.csv")

x = data.drop(columns=['Class'])
y = data['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=2)

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    scale_pos_weight=1,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=0
)

xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Classification Report:\n{report}')

conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')