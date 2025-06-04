import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


df_E = pd.read_csv("Clustering/0602/data/preprocessed_data/E/preprocessed_E.csv")
df_E = df_E[['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']].drop_duplicates().reset_index(drop=True)
df_F = pd.read_csv("Clustering/0602/data/preprocessed_data/F/preprocessed_F.csv")
df_F = df_F[['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']].drop_duplicates().reset_index(drop=True)
df_G = pd.read_csv("Clustering/0602/data/preprocessed_data/G/preprocessed_G.csv")
df_G = df_G[['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']].drop_duplicates().reset_index(drop=True)
df_H = pd.read_csv("Clustering/0602/data/preprocessed_data/H/preprocessed_H.csv")
df_H = df_H[['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']].drop_duplicates().reset_index(drop=True)
df = pd.concat([df_E, df_F, df_G, df_H], ignore_index=True).drop_duplicates()


X = df.drop(columns=['TRAVELER_ID'])

# elbow method  --> (클러스터 개수 8~12개가 적당해보임)
"""
inertia = []  # Within-Cluster Sum of Squares (클러스터 내 오차 제곱합)

for k in range(1, 50):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)  

plt.plot(range(1, 50), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()
"""

for k in range(8, 13):
    kmeans_final = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans_final.fit_predict(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=df['cluster'], cmap='tab10', alpha=0.7)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(f'K-means clustering result (PCA 2D) k={k}')
    plt.legend(*scatter.legend_elements(), title='Cluster')
    plt.show()

    df.to_csv(f'Clustering/0602/data/clustered_data/clustered_k={k}.csv', index=False)