import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples, pairwise_distances_argmin, davies_bouldin_score, calinski_harabasz_score

df = pd.read_csv("Clustering/0602/data/clustered_data/clustered_k=3_per_gender&age.csv")

gender_vals = [0, 1]
agegrp_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
combos = [(g, a) for g in gender_vals for a in agegrp_vals]


#클러스터링 결과 바탕으로 새로운 데이터 실험
"""
# 새 데이터 (TRAVEL_STYL_1, 3, 5, 6, 7, 8)
new_data = np.array([[0.7, 0.4, 0.8, 0.2, 0.9, 0.6]])

# 해당 그룹 (GENDER, AGE_GRP)의 데이터 필터링
g, a = 1, 0.75
subset = df[(df['GENDER'] == g) & (df['AGE_GRP'] == a)]

# 사용된 features
feature_cols = ['TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5',
                'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']

# 클러스터 중심값 계산
centroids = subset.groupby('CLUSTER')[feature_cols].mean().values

print(centroids)

# 새 데이터와 각 클러스터 중심 간 거리 비교 → 가장 가까운 클러스터 선택
predicted_cluster = pairwise_distances_argmin(new_data, centroids)[0]
print(f"새로운 데이터는 클러스터 {predicted_cluster}에 속하는 것으로 예측됩니다.")
vec1 = np.array([0.18905473, 0.17495854, 0.47678275, 0.41293532, 0.29187396, 0.83996683] )
distance = np.linalg.norm(vec1 - new_data)
print(f"Euclidean distance: {distance:.4f}")
"""

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

#2. 각 클러스터별 TRAVEL_STYL 점수 평균 막대그래프
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

#각 클러스터별 TRAVEL_STYL 점수 평균 레이더차트
"""
# 여행 성향 feature
features = ['TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5',
            'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']
num_features = len(features)
angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False).tolist()
angles += angles[:1]  # 닫기 위해 첫 점 추가

# 성별, 연령대 조합
gender_vals = [0, 1]
agegrp_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
combos = [(g, a) for g in gender_vals for a in agegrp_vals]

# 전체 subplot 생성
fig, axes = plt.subplots(2, 5, subplot_kw={'polar': True}, figsize=(15, 25))
fig.suptitle('Average of Travel Style (Radar Chart)', fontsize=18)

for idx, (g, a) in enumerate(combos):
    row = g  # 성별: 0이면 1행, 1이면 2행
    col = agegrp_vals.index(a)  # 열: 연령대 순서
    ax = axes[row, col]

    subset = df[(df['GENDER'] == g) & (df['AGE_GRP'] == a)]
    mean_df = subset.groupby('CLUSTER')[features].mean()

    for cluster_id, row in mean_df.iterrows():
        values = row.tolist()
        values += values[:1]  # 레이더 차트 닫기
        ax.plot(angles, values, label=f'Cluster {cluster_id}')
        ax.fill(angles, values, alpha=0.25)

    ax.set_title(f'GENDER={g}, AGE_GRP={a}', size=8)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=5)
    ax.set_yticklabels([])  # radial label은 숨김
    ax.legend(loc='upper right', fontsize=5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
"""



#클러스터별 여행스타일 평균, 분산 프린트
"""
gender_vals = [0, 1]
agegrp_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
combos = [(g, a) for g in gender_vals for a in agegrp_vals]

feature_cols = ['TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 
                'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']

for g, a in combos:
    subset = df[(df['GENDER'] == g) & (df['AGE_GRP'] == a)]
    print(f"\n===== GENDER={g}, AGE_GRP={a} =====")

    # 클러스터별 평균과 분산
    grouped = subset.groupby('CLUSTER')[feature_cols]
    mean_df = grouped.mean()
    var_df = grouped.var()

    for cluster in mean_df.index:
        print(f"\n-- CLUSTER {cluster} --")
        print("평균:")
        print(mean_df.loc[cluster])
        print("분산:")
        print(var_df.loc[cluster])
"""

#3. 클러스터 별 TRAVEL-STYL 변수들의 F값, p값 
"""
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
"""


#4. Silhouette Coefficient (전체 평균)
"""
features_cols = ['TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 
                 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']
sc = 0
print("Silhouette Coefficients per Group:\n")

for g, a in combos:
    subset = df[(df['GENDER'] == g) & (df['AGE_GRP'] == a)]
    
    X = subset[features_cols]
    labels = subset['CLUSTER']

    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels) 
    chi = calinski_harabasz_score(X, labels)  
    sc = sc + chi
    print(f"GENDER={g}, AGE_GRP={a}")
    print(f"Davies-Bouldin Index: {dbi:.4f}")        # 낮을수록 좋음
    print(f"Calinski-Harabasz Index: {chi:.4f}")     # 높을수록 좋음
    print(f"Silhouette Coefficient = {sil:.4f}\n")

print(sc/10)
"""


#5. Silhouette Coefficinet (클러스터별)
"""
features = ['TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5',
            'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']

# 결과 저장용 DataFrame
results = []

for g, a in combos:
    subset = df[(df['GENDER'] == g) & (df['AGE_GRP'] == a)]
    
    if subset.shape[0] < 2 or subset['CLUSTER'].nunique() < 2:
        print(f"GENDER={g}, AGE_GRP={a} → 데이터 부족 또는 클러스터 수 부족")
        continue

    X = subset[features]
    labels = subset['CLUSTER']
    
    sample_silhouette_vals = silhouette_samples(X, labels)

    subset = subset.copy()
    subset['silhouette'] = sample_silhouette_vals

    # 클러스터별 평균 실루엣 계수 계산
    cluster_means = subset.groupby('CLUSTER')['silhouette'].mean()

    for cluster_label, score in cluster_means.items():
        results.append({
            'GENDER': g,
            'AGE_GRP': a,
            'CLUSTER': cluster_label,
            'SILHOUETTE': score
        })

# 결과 DataFrame으로 변환
silhouette_df = pd.DataFrame(results)
print(silhouette_df)
"""

#6. 클러스터별 개수
for g, a in combos:
    subset = df[(df['GENDER'] == g) & (df['AGE_GRP'] == a)]
    print(f"GENDER={g}, AGE_GRP={a}")
    print(len(subset))
    #print(subset['CLUSTER'].value_counts().sort_index())
    print("-" * 30)