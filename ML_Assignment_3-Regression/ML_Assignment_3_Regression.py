# 1. Loading and Preprocessing

from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


# Load the dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target


# Check for missing values
missing = df.isnull().sum()

df.info()

df.head()

print(df.describe())

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""Converting the dataset to a pandas DataFrame makes it easier to handle, inspect, and manipulate.

It’s important to verify data quality before training models. Although this dataset has no missing values, checking ensures robustness.

Some regression models (e.g., SVR, Gradient Boosting) perform better when features are on a similar scale.

Standardization improves convergence in optimization and helps the model focus on feature relationships, not their magnitudes.
"""

# 2. Regression Algorithm Implementation

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#A. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

"""Linear Regression assumes a linear relationship between the input features and the target variable. It fits a straight line (or hyperplane in multidimensional space) to minimize the sum of squared residuals (errors).

Suitability:
This model serves as a good baseline. It's efficient and interpretable but may underperform on complex, non-linear data like housing prices which can depend on interactions between features.
"""

#B. Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

"""A decision tree splits the dataset into branches using feature thresholds that reduce variance in the target variable. It continues splitting until it reaches a stopping condition (e.g., max depth or minimum samples per leaf).

Suitability:
Captures non-linear relationships and interactions between features. However, it is prone to overfitting, especially if not properly pruned or regularized.
"""

#C. Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

"""Random Forest is an ensemble of multiple decision trees. Each tree is trained on a random subset of the data and features (bagging). Predictions are averaged across all trees to reduce variance.

Suitability:
More robust and accurate than a single decision tree. Handles non-linearity well and is less prone to overfitting due to averaging.
"""

#D. Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)

"""Gradient Boosting builds trees sequentially, where each new tree tries to correct the errors made by the previous one. The model is optimized using gradient descent.

Suitability:
Highly effective for structured/tabular data. Performs well with moderate-sized datasets and complex feature interactions. Often achieves top performance in regression tasks.
"""

#E. Support Vector Regressor
svr = SVR()
svr.fit(X_train, y_train)

"""SVR tries to find a function that fits the data within a specified margin of tolerance (epsilon). It uses kernel functions to model non-linear relationships and maximizes the margin around the function.

Suitability:
Works well with small to medium datasets and clean, scaled data. However, it can struggle with larger datasets like this one and is sensitive to scaling and parameter tuning.
"""

#3. Model Evaluation and Comparison

def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'Model': name, 'MSE': mse, 'MAE': mae, 'R2': r2}

results = []
models = [(lr, 'Linear Regression'), (dt, 'Decision Tree'), (rf, 'Random Forest'),
          (gbr, 'Gradient Boosting'), (svr, 'SVR')]

for model, name in models:
    results.append(evaluate_model(model, name))

results_df = pd.DataFrame(results)
results_df.sort_values('R2', ascending=False)

"""Best-Performing Algorithm: Gradient Boosting Regressor
Highest R² Score (0.82): Indicates the model explains the most variance in the target variable.
Lowest MSE and MAE: Shows it makes the most accurate predictions on average.

Worst-Performing Algorithm: Support Vector Regressor (SVR)
Lowest R² Score (0.48): Indicates poor explanation of variance in target.

Highest MSE and MAE: Larger prediction errors than other models.
"""

