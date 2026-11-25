"""
Week 12 - Data Splitting and K-Fold Cross-Validation (Lab VII, AI & ML)
"""

# ----------------------------------------------------------------------
# Q1 - Churn data: 70/30 train/test and 60/20/20 train/validation/test splits
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Activity 1.1: Data Preparation (simulated churn data)
np.random.seed(0)
churn_data = pd.DataFrame({
    'MonthlyCharges': np.random.uniform(20, 100, 200),
    'DataUsageGB': np.random.uniform(10, 80, 200),
    'ContractYears': np.random.randint(1, 4, 200),
    'Churn': np.random.choice([0, 1], 200, p=[0.7, 0.3])})
X = churn_data.drop('Churn', axis=1)
y = churn_data['Churn']
print("Original Churn Data (first 5 rows):\n", churn_data.head())

# Activity 1.2: Train-Test Split (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
print("\nTrain-Test split (70/30):")
print("  X_train shape:", X_train.shape, " y_train shape:", y_train.shape)
print("  X_test  shape:", X_test.shape, " y_test  shape:", y_test.shape)

# Activity 1.3: Train-Validation-Test Split (60/20/20, two splits)
X_train2, X_temp, y_train2, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42)
X_val, X_test2, y_val, y_test2 = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)
print("\nTrain-Validation-Test split (60/20/20):")
print("  X_train shape:", X_train2.shape)
print("  X_val   shape:", X_val.shape)
print("  X_test  shape:", X_test2.shape)

# ----------------------------------------------------------------------
# Q2 - 5-fold cross-validation with logistic regression
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

np.random.seed(0)
churn_data = pd.DataFrame({
    'MonthlyCharges': np.random.uniform(20, 100, 200),
    'DataUsageGB': np.random.uniform(10, 80, 200),
    'ContractYears': np.random.randint(1, 4, 200),
    'Churn': np.random.choice([0, 1], 200, p=[0.7, 0.3])})
X = churn_data.drop('Churn', axis=1)
y = churn_data['Churn']

# Activity 2.1: K-Fold Implementation (5 folds)
scores = cross_val_score(LogisticRegression(max_iter=1000), X, y,
                         cv=KFold(n_splits=5))
for i, s in enumerate(scores, 1):
    print(f"Fold {i} accuracy: {s:.4f}")
print("\nAverage accuracy across 5 folds:", round(scores.mean(), 4))
