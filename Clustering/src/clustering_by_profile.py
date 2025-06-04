import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

region = 'F'

df = pd.read_csv(f"data/preprocessed_data/{region}/preprocessed_traveller_master_{region}.csv")

X = df.drop(columns=['TRAVELER_ID']) 

# elbow method로 최적의 클러스터 개수 구하기 (=>6)
"""
inertia = []
K_range = range(1, 30)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()
"""

kmeans = KMeans(n_clusters=6, random_state=42)
df['CLUSTER'] = kmeans.fit_predict(X)

print("클러스터링 완료")

# 클러스터별 평균값 확인
"""
df.groupby('CLUSTER').mean(numeric_only=True)
"""

# PCA로 2차원 분포 확인
"""
X = df.drop(columns=['TRAVELER_ID', 'CLUSTER'])

# PCA로 2차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 시각화
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['CLUSTER'], cmap='viridis', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Visualization of Clusters')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()
"""

