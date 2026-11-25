"""
Week 6 - Feature Selection, Normalization, and Discretization (Lab VII, AI & ML)
"""

# ----------------------------------------------------------------------
# Q1 - California Housing: select 5 features
# ----------------------------------------------------------------------
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame
selected = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']]
print(selected.head())
print("Before:", df.shape, "-> After:", selected.shape)

# ----------------------------------------------------------------------
# Q2 - Iris: Min-Max normalization and Z-score standardization
# ----------------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

X = pd.DataFrame(load_iris().data,
                 columns=['sepal_l','sepal_w','petal_l','petal_w'])

X_mm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
X_z  = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
print(X_mm.head(2).round(2))
print(X_z.head(2).round(2))

# ----------------------------------------------------------------------
# Q3 - Adult Income: age bins with pd.cut (17-30 / 31-45 / 46-60 / 61+)
# ----------------------------------------------------------------------
import pandas as pd

df = pd.read_csv('adult.data', header=None)
df.columns = ['age','workclass','fnlwgt','education','edu_num','marital',
              'occupation','relation','race','sex','gain','loss',
              'hours','country','income']

df['age_group'] = pd.cut(df['age'],
                         bins=[17, 30, 45, 60, 100],
                         labels=['Young Adult','Middle-Aged',
                                 'Senior Adult','Elderly'])
print(df['age_group'].value_counts())
