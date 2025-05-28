
ML-Assignment-4-Classification Problem

# 1. Loading and Preprocessing

from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X.info()

X.head()

X.describe()

# Check for missing values
print(X.isnull().sum().sum())

"""Missing values: The dataset contains no missing values, as confirmed above.

Feature Scaling: Most ML models perform better when features are on a similar scale. Especially important for SVM and k-NN.
"""

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Classification Algorithm Implementation

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Logistic Regression
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))

"""Logistic Regression models the probability of the default class (malignant or benign) using a logistic function. It's fast and performs well on linearly separable data."""

# 2. Decision Tree Classifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_preds))

"""A tree-based model that splits features based on criteria like Gini or entropy. Easy to interpret but can overfit on training data"""

# 3. Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))

# 4. Support Vector Machine (SVM)
svm = SVC()
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))

"""SVM finds the optimal hyperplane that separates classes. It's effective in high-dimensional spaces, especially with proper scaling."""

# 5. k-Nearest Neighbors (k-NN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
knn_preds = knn.predict(X_test_scaled)
print("k-NN Accuracy:", accuracy_score(y_test, knn_preds))

"""k-NN is a non-parametric method that predicts the label based on the majority class of nearest neighbors. It's simple but computationally intensive."""

# 3. Model Comparison

results = {
    'Logistic Regression': accuracy_score(y_test, lr_preds),
    'Decision Tree': accuracy_score(y_test, dt_preds),
    'Random Forest': accuracy_score(y_test, rf_preds),
    'SVM': accuracy_score(y_test, svm_preds),
    'k-NN': accuracy_score(y_test, knn_preds)
}

for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

"""Best: Likely Random Forest or SVM, both known to perform well on this dataset.

Worst: Potentially Decision Tree due to overfitting
"""

