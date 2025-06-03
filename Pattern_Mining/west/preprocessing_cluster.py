import pandas as pd

df1 = pd.read_csv("preprocessed_travel_data.csv")
df2 = pd.read_csv("preprocessed_with_cluster_G.csv")

cols1 = ['TRAVELER_ID','GENDER','AGE_GRP','TRAVEL_STATUS_ACCOMPANY']
new_df1 = df1[cols1].copy()

cols2 = ['TRAVELER_ID','VISIT_AREA_NM', 'VISIT_AREA_TYPE_CD', 'CLUSTER']
new_df2 = df2[cols2].copy()

merged = pd.merge(
    new_df1, 
    new_df2, 
    on='TRAVELER_ID', 
    how='inner'
)
merged = merged.drop_duplicates()

merged.to_csv("preprocessed_travel_data_final.csv", index=False)
