
MODULE END ASSIGNMENT_ML

#1. Loading and Preprocessing

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('CarPrice_Assignment.csv')

print(df.shape)

df.head()

df.info()
df.describe()

df['fueltype'] = df['fueltype'].str.lower()
df['carbody'] = df['carbody'].str.lower()

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Check for nulls
print(df.isnull().sum().sum())

# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('price', axis=1))
X = pd.DataFrame(scaled_features, columns=df.drop('price', axis=1).columns)
y = df['price']

# 2. Model Implementation

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "R2": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred)
    }

results_df = pd.DataFrame(results).T
results_df

# 3. Model Evaluation

# Visual comparison
results_df.plot(kind='bar', figsize=(12,6), title="Model Comparison")
plt.ylabel("Score / Error")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

"""Random Forest Regressor had the highest R² and lowest MSE and MAE, making it the most reliable model for price prediction."""

# 4. Feature Importance Analysis

# For Random Forest
importances = models["Random Forest"].feature_importances_
feature_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
feature_imp[:10].plot(kind='bar', title='Top 10 Important Features', figsize=(10,6))
plt.show()

"""Features like enginesize, curbweight, and horsepower are top contributors to price."""

#5. Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test)

print("R² after tuning:", r2_score(y_test, y_pred_tuned))
print("MSE after tuning:", mean_squared_error(y_test, y_pred_tuned))
print("MAE after tuning:", mean_absolute_error(y_test, y_pred_tuned))

