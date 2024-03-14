import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('your_dataset.csv')

# Data preprocessing
# Assume the target variable is named 'Target' and ID column is 'ID'
target = 'Target'
IDcol = 'ID'

# Fill missing values
df.fillna(-999, inplace=True)

# Convert categorical variables to numerical format if any
df = pd.get_dummies(df)

# Split data into features and target
X = df.drop([target, IDcol], axis=1)
y = df[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model fitting
model = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# Evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Parameter tuning for 'max_depth' and 'min_child_weight'
param_test1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}

gsearch1 = GridSearchCV(
    estimator=XGBClassifier(
        learning_rate=0.1, n_estimators=140, max_depth=5,
        min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27
    ), 
    param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5
)
gsearch1.fit(X_train, y_train)
print(gsearch1.best_params_)

# Parameter tuning for 'gamma'
param_test3 = {
    'gamma': [i/10.0 for i in range(0, 5)]
}

gsearch3 = GridSearchCV(
    estimator=XGBClassifier(
        learning_rate=0.1, n_estimators=140, max_depth=gsearch1.best_params_['max_depth'],
        min_child_weight=gsearch1.best_params_['min_child_weight'], gamma=0, subsample=0.8,
        colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27
    ),
    param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5
)
gsearch3.fit(X_train, y_train)
print(gsearch3.best_params_)

# Final model retraining with tuned parameters
final_model = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=gsearch1.best_params_['max_depth'],
    min_child_weight=gsearch1.best_params_['min_child_weight'],
    gamma=gsearch3.best_params_['gamma'],
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27
)

final_model.fit(X_train, y_train)
y_pred_final = final_model.predict(X_test)
predictions_final = [round(value) for value in y_pred_final]
accuracy_final = accuracy_score(y_test, predictions_final)
print("Final Accuracy: %.2f%%" % (accuracy_final * 100.0))

# Feature importance
plt.figure(figsize=(10, 8))
plt.bar(range(len(final_model.feature_importances_)), final_model.feature_importances_)
plt.show()
