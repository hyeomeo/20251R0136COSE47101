from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

region = 'H'

final_df = pd.read_csv(f"data/preprocessed_data/{region}/preprocessed_visit_area_info_{region}.csv")
X = final_df.drop(columns=['TRAVELER_ID'])

# elbow method로 최적의 클러스터 개수 구하기 (=>6)
"""
inertia = []  # Within-Cluster Sum of Squares (클러스터 내 오차 제곱합)

for k in range(1, 30):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)  

plt.plot(range(1, 30), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()
"""

optimal_k = 6  
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
final_df['cluster'] = kmeans_final.fit_predict(X)

# 여행지 타입 별 클러스터별 평균 방문 횟수 그래프(바 형태)
"""
visit_cols = [f'AVG_VISIT_PER_TRIP_TYPE_{i}' for i in range(1,14)]

cluster_means = final_df.groupby('cluster')[visit_cols].mean()

cluster_means.T.plot(kind='bar', figsize=(12,6))
plt.title('Avg Num of Visits per Visit Area Type (1~13)')
plt.xlabel('Visit Area Type')
plt.ylabel('Avg Num of Visits')
plt.legend(title='Cluster')
plt.show()
"""

# PCA로 2차원 분포 확인
"""
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=final_df['cluster'], cmap='tab10', alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('K-means clustering result (PCA 2D)')
plt.legend(*scatter.legend_elements(), title='Cluster')
plt.show()
"""

# 여행지 타입 별 클러스터별 평균 방문 횟수, 평균 만족도
"""
visit_cols = [f'AVG_VISIT_PER_TRIP_TYPE_{i}' for i in range(1,14)]
dgstfn_cols = [f'AVG_DGSTFN_TYPE_{i}' for i in range(1,14)]

visit_means = final_df.groupby('cluster')[visit_cols].mean()
dgstfn_means = final_df.groupby('cluster')[dgstfn_cols].mean()

visit_means.index = [f'Cluster {i} - Visit' for i in visit_means.index]
dgstfn_means.index = [f'Cluster {i} - DGSTFN' for i in dgstfn_means.index]

# Combine into one heatmap dataframe
heatmap_data = pd.concat([visit_means, dgstfn_means])

plt.figure(figsize=(18, 8))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={'label': 'Mean Value'})
plt.title('Cluster-wise Mean Visit Frequency and Satisfaction (Types 1–13)')
plt.xlabel('Type')
plt.ylabel('Cluster - Metric')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""

final_df.to_csv(f"data/result_data/clustered_by_visits_{region}.csv", index=False)

print("클러스터링 완료")