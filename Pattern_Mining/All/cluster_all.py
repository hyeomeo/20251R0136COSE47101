import pandas as pd

df1 = pd.read_csv("travel_data_merged.csv")
df2 = pd.read_csv("updated_clustered_data_total_k30.csv")

cols1 = ['TRAVELER_ID','GENDER','AGE_GRP','TRAVEL_STATUS_ACCOMPANY', 'VISIT_AREA_NM']
new_df1 = df1[cols1].copy()

cols2 = ['TRAVELER_ID', 'CLUSTER_NEW']
new_df2 = df2[cols2].copy()

merged = pd.merge(
    new_df1, 
    new_df2, 
    on='TRAVELER_ID', 
    how='inner'
)
merged = merged.drop_duplicates()

merged.to_csv("travel_data_final.csv", index=False)