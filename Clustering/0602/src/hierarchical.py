# 계층적 클러스터링 (성별, 연령대 별 진행)
# linkage, n_cluster을 바꿔가며 여러번 시도해보았으나 k means 보다 실루엣 계수가 낮거나 클러스터 간 데이터 수 불균형 심각

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
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

linkage_options = ['ward', 'complete', 'average', 'single']
n_clusters_range = range(2, 10)  

for g, a in combos:
    print(f"\nGENDER={g}, AGE_GRP={a}")
    subset = df[(df['GENDER'] == g) & (df['AGE_GRP'] == a)]
    
    X = subset[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for linkage in linkage_options:
        for n_clusters in n_clusters_range:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = model.fit_predict(X_scaled)
            if len(set(labels)) < 2:
                continue  
            score = silhouette_score(X_scaled, labels)
            if score < 0.15:
                continue
            subset = subset.copy()
            subset['CLUSTER'] = labels
            print("클러스터별 개수:")
            print(subset['CLUSTER'].value_counts().sort_index())
            print(f'Silhouette Score: {score:.4f}')
   