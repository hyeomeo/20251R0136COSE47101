# DBSCAN (성별, 연령대 별 진행)
# k distance plot(elbow랑 비슷한 접근법) 등의 방법으로 eps, min_sample을 여러값으로 시도해봤지만 실루엣계수가 k means보다 작게 나옴

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

eps_values = np.linspace(0.5, 3.0, 6) # 0.5부터 3.0까지 6개
min_samples_values = [5, 10, 15, 20, 25]

for i, (g, a) in enumerate(combos, 1):
    print(f"GENDER={g}, AGE_GRP={a}")
    subset_idx = df[(df['GENDER'] == g) & (df['AGE_GRP'] == a)].index
    subset = df.loc[subset_idx]
    X = subset[features]

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for eps in eps_values:
        for min_samples in min_samples_values:
            # DBSCAN 클러스터링
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
            # 실루엣 계수 계산
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters >= 2:
                score = silhouette_score(X_scaled, labels)
                print(f"eps={eps}, min_samples={min_samples}")
                print(f"Silhouette Score: {score:.4f} (클러스터 수: {n_clusters})")
            else:
                continue
            