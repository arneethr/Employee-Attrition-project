import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint

# Load dataset
df = pd.read_csv(r"HR-Employee-Attrition.csv")

# Step 1: Data Preprocessing
df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)

# Encoding categorical columns
categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop('Attrition', axis=1)
y = le.fit_transform(df['Attrition'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Hyperparameter Tuning using RandomizedSearchCV
# Define the parameter grid
param_dist = {
    'n_estimators': randint(50, 200),              # Number of trees in the forest
    'max_depth': randint(5, 20),                   # Maximum depth of the tree
    'min_samples_split': randint(2, 10),           # Minimum samples required to split a node
    'min_samples_leaf': randint(1, 4),             # Minimum samples required at each leaf node
    'max_features': ['sqrt', 'log2', None]       # Number of features to consider when looking for best split
}

# Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, 
    n_iter=50,            # Number of iterations for random search
    cv=5,                 # 5-fold cross-validation
    scoring='accuracy',    # Use accuracy as the scoring metric
    random_state=42,       # For reproducibility
    n_jobs=-1             # Use all available cores
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Best hyperparameters
print(f"Best Hyperparameters: {random_search.best_params_}")

# Step 3: Train the optimized model
best_rf = random_search.best_estimator_
best_rf.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = best_rf.predict(X_test)

# Performance metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Accuracy: {accuracy}")

report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{report}")

conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Step 5: Feature Importance
feature_importance = best_rf.feature_importances_
important_features = pd.Series(feature_importance, index=X.columns).sort_values(ascending=False)
print("Optimized Feature Importance:\n", important_features)
