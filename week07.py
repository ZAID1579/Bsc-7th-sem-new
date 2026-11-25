"""
Week 7 - Supervised Learning: KNN and Linear Regression (Lab VII, AI & ML)
"""

# ----------------------------------------------------------------------
# Q1 - Fruit classifier with KNN (k=3, 70/30 split, random_state=42)
# ----------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.DataFrame({
    'Size_cm':         [7, 8, 6, 9, 12, 13, 11, 8, 7, 12],
    'Color_Intensity': [8, 7, 9, 6,  4,  5,  3, 8, 9,  4],
    'Fruit': ['Apple','Apple','Apple','Orange','Banana',
              'Banana','Banana','Orange','Apple','Banana']})
features = ['Size_cm', 'Color_Intensity']

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df['Fruit'], test_size=0.3, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

print("Accuracy:", round(model.score(X_test, y_test), 2))
new = pd.DataFrame([[7.5, 8]], columns=features)
print("New fruit is:", model.predict(new)[0])

# ----------------------------------------------------------------------
# Q2 - Linear regression: temperature vs electricity, MSE + scatter plot
# ----------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

temp = np.array([10, 14, 18, 22, 26, 30, 32]).reshape(-1, 1)
kwh  = np.array([50, 58, 66, 78, 90, 102, 110])

model = LinearRegression()
model.fit(temp, kwh)
pred = model.predict(temp)

print("MSE:", round(mean_squared_error(kwh, pred), 2))

plt.scatter(temp, kwh)
plt.plot(temp, pred, color='red')
plt.xlabel('Temperature (C)'); plt.ylabel('Electricity (kWh)')
plt.title('Temperature vs Electricity Use')
plt.savefig('temp_vs_kwh.png')
