# k-means 클러스터링 (성별, 연령대 별 진행)
 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


df_E = pd.read_csv("Clustering/0602/data/preprocessed_data/E/preprocessed_E.csv")
df_E = df_E[['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']].drop_duplicates().reset_index(drop=True)
df_F = pd.read_csv("Clustering/0602/data/preprocessed_data/F/preprocessed_F.csv")
df_F = df_F[['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']].drop_duplicates().reset_index(drop=True)
df_G = pd.read_csv("Clustering/0602/data/preprocessed_data/G/preprocessed_G.csv")
df_G = df_G[['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']].drop_duplicates().reset_index(drop=True)
df_H = pd.read_csv("Clustering/0602/data/preprocessed_data/H/preprocessed_H.csv")
df_H = df_H[['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']].drop_duplicates().reset_index(drop=True)
df = pd.concat([df_E, df_F, df_G, df_H], ignore_index=True).drop_duplicates()


gender_vals = [0, 1]
agegrp_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
combos = [(g, a) for g in gender_vals for a in agegrp_vals]

features = ['TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']

score_total = 0

plt.figure(figsize=(15, 20))
for i, (g, a) in enumerate(combos, 1):
    print(f"GENDER={g}, AGE_GRP={a}")
    subset_idx = df[(df['GENDER'] == g) & (df['AGE_GRP'] == a)].index
    subset = df.loc[subset_idx]
    X = subset[features]
    
    # k-means 클러스터링
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    df.loc[subset_idx, 'CLUSTER'] = labels

    # silhouette score 계산
    score = silhouette_score(X, labels) # k를 얼마로 하던 0.15 근처의 값이 나옴..
    score_total = score_total + score
    print(f"Silhouette Score: {score:.4f}")
    
    #시각화
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    plt.subplot(5, 2, i)
    scatter = plt.scatter(components[:, 0], components[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(f'GENDER={g}, AGE_GRP={a}')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.colorbar(scatter, ticks=range(3), label='Cluster')
    plt.tight_layout()

plt.show()

#df.to_csv(f'Clustering/0602/data/clustered_data/clustered_k={k}_per_gender&age.csv', index=False)

print(score_total/10) # 평균 실루엣 계수