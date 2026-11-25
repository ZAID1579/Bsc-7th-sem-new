"""
Week 5 - Data Cleaning and Integration (Lab VII, AI & ML)
"""

# ----------------------------------------------------------------------
# Q1 - Titanic: drop Cabin, fill Age with median, Embarked with mode
# ----------------------------------------------------------------------
import pandas as pd

df = pd.read_csv('train.csv')
print("Shape:", df.shape)
print(df.isnull().sum())

df = df.drop(columns=['Cabin'])

median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)

mode_emb = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode_emb)

print(df.isnull().sum())

# ----------------------------------------------------------------------
# Q2 - Employee data: IQR outlier detection and capping
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np

np.random.seed(42)
scores = np.random.normal(75, 10, 100)
scores[10], scores[25], scores[50] = 150, -20, 5
df = pd.DataFrame({'ID': range(1, 101), 'Score': scores.round(0)})

Q1 = df['Score'].quantile(0.25)
Q3 = df['Score'].quantile(0.75)
IQR = Q3 - Q1
low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR

out = df[(df['Score'] < low) | (df['Score'] > high)]
print(out[['ID', 'Score']])

df.loc[df['Score'] > high, 'Score'] = high
df.loc[df['Score'] < low, 'Score'] = low
print("Capped. Nothing deleted.")
