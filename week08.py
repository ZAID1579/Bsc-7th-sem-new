"""
Week 8 - Decision Trees and Logistic Regression (Lab VII, AI & ML)
"""

# ----------------------------------------------------------------------
# Q1 - Decision Tree: customer campaign response (max_depth=3)
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
data = {'Age': np.random.randint(20, 60, 100),
        'PurchaseFrequency': np.random.uniform(0.5, 5, 100).round(1),
        'RespondedToCampaign': np.random.choice([0, 1], 100, p=[0.6, 0.4])}
for i in range(100):
    if data['Age'][i] < 35 and data['PurchaseFrequency'][i] > 3:
        data['RespondedToCampaign'][i] = 1
    elif data['Age'][i] > 50 and data['PurchaseFrequency'][i] < 1.5:
        data['RespondedToCampaign'][i] = 0

df = pd.DataFrame(data)
X = df[['Age', 'PurchaseFrequency']]
y = df['RespondedToCampaign']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nCampaign Response Data (X_train head):\n", X_train.head())
print("\nCampaign Response Target (y_train head):\n", y_train.head())

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred))

# ----------------------------------------------------------------------
# Q2 - Logistic Regression: email spam detection
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Activity 2.1: Data Preparation
np.random.seed(42)
keywords = np.random.randint(0, 10, 100)        # Number of suspicious keywords
spam_prob = 1 / (1 + np.exp(-(keywords - 5)))   # Sigmoid-like function
is_spam = (np.random.rand(100) < spam_prob).astype(int)

df = pd.DataFrame({'NumSuspiciousKeywords': keywords, 'IsSpam': is_spam})
X = df[['NumSuspiciousKeywords']]   # Features must be 2D
y = df['IsSpam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Spam Detection Data (X_train head):\n", X_train.head())
print("\nSpam Target (y_train head):\n", y_train.head())

# Activity 2.2: Logistic Regression Training
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nLogistic Regression Accuracy:", accuracy)
