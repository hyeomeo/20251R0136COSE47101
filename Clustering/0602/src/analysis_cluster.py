import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("Clustering/0602/data/clustered_data/clustered_k=3_per_gender&age.csv")

gender_vals = [0, 1]
agegrp_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
combos = [(g, a) for g in gender_vals for a in agegrp_vals]

#1. pca를 이용한 2차원 시각화
"""
plt.figure(figsize=(15, 20))
for i, (g, a) in enumerate(combos, 1):
    subset_idx = df[(df['GENDER'] == g) & (df['AGE_GRP'] == a)].index
    subset = df.loc[subset_idx]
    features = subset[['TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']]
    cluster_labels = subset['CLUSTER'].values

    #시각화
    pca = PCA(n_components=2)
    components = pca.fit_transform(features)

    # 시각화
    plt.subplot(5, 2, i)
    scatter = plt.scatter(components[:, 0], components[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    plt.title(f'GENDER={g}, AGE_GRP={a}')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.colorbar(scatter, ticks=range(3), label='Cluster')
    plt.tight_layout()

    

plt.show()
"""

#2. 각 클러스터별 TRAVEL_STYL 점수 평균
"""
features = ['TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 
            'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']

for g, a in combos:
    subset = df[(df['GENDER'] == g) & (df['AGE_GRP'] == a)]
    if subset.empty:
        continue

    # 클러스터별 평균 계산
    cluster_means = subset.groupby('CLUSTER')[features].mean()
    cluster_means = cluster_means.T  # transpose for plotting (features x clusters)

    # bar plot 그리기
    x = range(len(features))
    width = 0.2  # 막대 너비
    num_clusters = cluster_means.shape[1]
    
    plt.figure(figsize=(10, 6))
    for i, cluster_id in enumerate(cluster_means.columns):
        values = cluster_means[cluster_id].values
        plt.bar([xi + i * width for xi in x], values, width=width, label=f'Cluster {cluster_id}')

    plt.xticks([xi + width * (num_clusters - 1) / 2 for xi in x], features, rotation=45)
    plt.xlabel('Feature')
    plt.ylabel('Mean Value')
    plt.title(f'Cluster-wise Mean Features (GENDER={g}, AGE_GRP={a})')
    plt.legend()
    plt.tight_layout()
    plt.show()
"""



#3. 클러스터 별 TRAVEL-STYL 변수들의 F값, p값 
from sklearn.feature_selection import f_classif

for i, (g, a) in enumerate(combos, 1):
    subset = df[(df['GENDER'] == g) & (df['AGE_GRP'] == a)]
    if subset.empty:
        continue

    features = ['TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 
                'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']
    X = subset[features]
    y = subset['CLUSTER']
    
    f_vals, p_vals = f_classif(X, y)
    print(f"\n=== GENDER={g}, AGE_GRP={a} ===")
    for f, p, name in sorted(zip(f_vals, p_vals, features), reverse=True):
        print(f"{name}: F = {f:.2f}, p = {p:.4f}")
