"""
Week 10 - Unsupervised Learning: K-Means and Hierarchical Clustering (Lab VII, AI & ML)
"""

# ----------------------------------------------------------------------
# Q1 - K-Means clustering of animals (k=3)
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Activity 1.1: Data Preparation (3 groups: Small, Medium, Large)
np.random.seed(42)
small_animals = np.random.normal(loc=[5, 20], scale=[1, 5], size=(20, 2))
medium_animals = np.random.normal(loc=[50, 80], scale=[10, 15], size=(20, 2))
large_animals = np.random.normal(loc=[300, 150], scale=[50, 25], size=(20, 2))
X_animals = pd.DataFrame(
    np.vstack([small_animals, medium_animals, large_animals]),
    columns=['Weight_kg', 'Height_cm'])

# Scale the data (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_animals)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_animals.columns)
print("Scaled Animal Data (first 5 rows):\n", X_scaled_df.head())

# Activity 1.2: K-Means Clustering Training
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled_df)
X_animals['Cluster'] = kmeans.labels_
print("\nCluster sizes:\n", X_animals['Cluster'].value_counts().sort_index())
centres = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nCluster centres (Weight_kg, Height_cm):")
for i, c in enumerate(centres):
    print(f"  Cluster {i}: Weight {c[0]:.1f} kg, Height {c[1]:.1f} cm")

# Activity 1.3: Visualization
plt.figure(figsize=(7, 5))
for c in range(3):
    pts = X_animals[X_animals['Cluster'] == c]
    plt.scatter(pts['Weight_kg'], pts['Height_cm'], label=f"Cluster {c}")
plt.scatter(centres[:, 0], centres[:, 1], marker="X", s=150,
            color="black", label="Centroids")
plt.xlabel("Weight_kg"); plt.ylabel("Height_cm")
plt.title("K-Means Clustering of Animals (k=3)")
plt.legend(); plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------
# Q2 - Hierarchical clustering of student study habits + dendrogram
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Activity 2.1: Data Preparation
np.random.seed(42)
X_students = pd.DataFrame({
    'Study_Hours_Week': np.random.randint(5, 40, 50),
    'Sleep_Hours_Night': np.random.uniform(4, 9, 50).round(1)})
# Add some patterns: high study, low sleep / low study, high sleep
X_students.loc[X_students['Study_Hours_Week'] > 30, 'Sleep_Hours_Night'] = \
    np.random.uniform(4, 6, sum(X_students['Study_Hours_Week'] > 30)).round(1)
X_students.loc[X_students['Study_Hours_Week'] < 10, 'Sleep_Hours_Night'] = \
    np.random.uniform(7, 9, sum(X_students['Study_Hours_Week'] < 10)).round(1)
print("Student Habit Data (first 5 rows):\n", X_students.head())

# Activity 2.2: Hierarchical Clustering (Ward linkage)
Z = linkage(X_students, method='ward')

# Activity 2.3: Dendrogram
plt.figure(figsize=(8, 4.5))
dendrogram(Z)
plt.title("Dendrogram - Student Study Habits (Ward linkage)")
plt.xlabel("Student"); plt.ylabel("Distance")
plt.tight_layout(); plt.show()

groups = fcluster(Z, t=2, criterion='maxclust')
X_students['Group'] = groups
print("\nTwo groups after cutting the dendrogram:")
print(X_students.groupby('Group')[['Study_Hours_Week',
                                    'Sleep_Hours_Night']].mean().round(1))
print("\nGroup sizes:\n", X_students['Group'].value_counts().sort_index())
