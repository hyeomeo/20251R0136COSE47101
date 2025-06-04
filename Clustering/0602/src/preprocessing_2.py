import pandas as pd
import numpy as np

region = 'H' #지역설정

df1 = pd.read_csv(f'Clustering/0602/data/preprocessed_data/{region}/preprocessed_{region}.csv')
df2 = pd.read_csv(f'Clustering/0602/data/clustered_data/clustered_k=3_per_gender&age.csv')
df2 = df2[['TRAVELER_ID', 'CLUSTER']]

df_clustered = df1.merge(df2, on='TRAVELER_ID', how='left')
df_clustered.to_csv(f'Clustering/0602/data/preprocessed_data/{region}/preprocessed_with_cluster_{region}.csv', index=False)

