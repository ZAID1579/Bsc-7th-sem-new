"""
Week 9 - Artificial Neural Networks: Perceptron and MLP (Lab VII, AI & ML)
"""

# ----------------------------------------------------------------------
# Q1 - Perceptron for the AND gate
# ----------------------------------------------------------------------
import numpy as np

# Activity 1.1: Data Preparation (AND gate)
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])   # AND gate outputs

# Activity 1.2: Simplified Perceptron Implementation
np.random.seed(42)
weights = np.random.rand(2) * 0.1
bias = np.random.rand(1)[0] * 0.1

def step_function(z):
    return 1 if z >= 0 else 0

learning_rate = 0.1
epochs = 20
for epoch in range(epochs):
    for x, y_true in zip(X_and, y_and):
        z = np.dot(x, weights) + bias
        prediction = step_function(z)
        error = y_true - prediction
        weights = weights + learning_rate * error * x
        bias = bias + learning_rate * error

print("Final weights:", weights)
print("Final bias:", bias)
print("Predictions for AND gate inputs:")
for x, y_true in zip(X_and, y_and):
    print(x, "->", step_function(np.dot(x, weights) + bias),
          "(expected", y_true, ")")

# ----------------------------------------------------------------------
# Q2 - MLP for shape classification (squares vs circles)
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Activity 2.1: Data Preparation (NumCorners, AspectRatio; 0 = Square, 1 = Circle)
X_shapes = np.array([
    [4, 1.0], [4, 1.05], [4, 0.95],   # Squares
    [0, 1.0], [0, 1.1], [0, 0.9],     # Circles
    [4, 0.5], [0, 2.0]                # Noisy points
])
y_shapes = np.array([0, 0, 0, 1, 1, 1, 0, 1])

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_shapes, y_shapes, test_size=0.3, random_state=42)
print("Shape Classification Data (X_train_s):\n", X_train_s)
print("\nShape Target (y_train_s):\n", y_train_s)

# Activity 2.2: Training an MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=500,
                    random_state=42, solver='adam')
mlp.fit(X_train_s, y_train_s)
y_pred_s = mlp.predict(X_test_s)
print("\nPredictions:", y_pred_s)
print("True labels:", y_test_s)
print("\nMLP Accuracy:", accuracy_score(y_test_s, y_pred_s))
