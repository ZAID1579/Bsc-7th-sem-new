"""
Week 11 - Underfitting, Overfitting, and Data Separability (Lab VII, AI & ML)
"""

# ----------------------------------------------------------------------
# Q1 - Noisy sine wave: linear (underfit) vs degree-20 polynomial (overfit)
# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Activity 1.1: Data Preparation (noisy sine wave)
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.5, 80)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Activity 1.2: Underfitting a Linear Model
lin = LinearRegression().fit(X_train, y_train)
print("Linear model (underfitting):")
print("  Train MSE:", round(mean_squared_error(y_train, lin.predict(X_train)), 4))
print("  Test  MSE:", round(mean_squared_error(y_test, lin.predict(X_test)), 4))

# Activity 1.3: Overfitting a High-Degree Polynomial Model
poly = make_pipeline(PolynomialFeatures(degree=20), LinearRegression())
poly.fit(X_train, y_train)
print("Degree-20 polynomial (overfitting):")
print("  Train MSE:", round(mean_squared_error(y_train, poly.predict(X_train)), 4))
print("  Test  MSE:", round(mean_squared_error(y_test, poly.predict(X_test)), 4))

X_plot = np.linspace(0, 5, 200).reshape(-1, 1)
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, label='Training Data')
plt.scatter(X_test, y_test, label='Testing Data', alpha=0.6)
plt.plot(X_plot, lin.predict(X_plot), label='Linear (degree 1)')
plt.plot(X_plot, poly.predict(X_plot), label='Polynomial (degree 20)')
plt.title("Underfitting vs Overfitting - Noisy Sine Wave")
plt.xlabel("X"); plt.ylabel("y"); plt.legend()
plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------
# Q2 - Data separability: logistic regression on separable vs circular data
# ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.linear_model import LogisticRegression

# Activity 2.1: Linearly Separable Data
X_linear, y_linear = make_classification(
    n_samples=100, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, random_state=4, class_sep=1.5)
lr1 = LogisticRegression(max_iter=1000).fit(X_linear, y_linear)
print("Linearly separable data - Logistic Regression accuracy:",
      round(lr1.score(X_linear, y_linear), 4))

# Activity 2.2: Non-linearly Separable Data (concentric circles)
X_nonlinear, y_nonlinear = make_circles(
    n_samples=100, noise=0.1, factor=0.5, random_state=1)
lr2 = LogisticRegression(max_iter=1000).fit(X_nonlinear, y_nonlinear)
print("Non-linearly separable data (circles) - Logistic Regression accuracy:",
      round(lr2.score(X_nonlinear, y_nonlinear), 4))

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].scatter(X_linear[:, 0], X_linear[:, 1], c=y_linear, cmap='viridis', s=40)
ax[0].set_title("Linearly Separable Data")
ax[0].set_xlabel("Feature 1"); ax[0].set_ylabel("Feature 2")
ax[1].scatter(X_nonlinear[:, 0], X_nonlinear[:, 1], c=y_nonlinear,
              cmap='plasma', s=40)
ax[1].set_title("Non-linearly Separable Data")
ax[1].set_xlabel("Feature 1"); ax[1].set_ylabel("Feature 2")
plt.tight_layout(); plt.show()
