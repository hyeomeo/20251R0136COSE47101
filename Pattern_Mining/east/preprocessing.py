import pandas as pd

df1 = pd.read_csv("tn_traveller_master_여행객 Master_F.csv")
df2 = pd.read_csv("cleaned_dataset.csv")
df3 = pd.read_csv("tn_travel_여행_F.csv")

cols1 = ['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STATUS_ACCOMPANY','TRAVEL_STATUS_YMD']
new_df1 = df1[cols1].copy()
new_df1['TRAVEL_STATUS_YMD'] = new_df1['TRAVEL_STATUS_YMD'].str[5:7]

cols2 = ['TRAVELER_ID', 'VISIT_AREA_NM']
new_df2 = df2[cols2].copy()
new_df2['TRAVELER_ID'] = new_df2['TRAVELER_ID'].str[2:]

cols3 = ['TRAVELER_ID', 'TRAVEL_PERSONA']
new_df3 = df3[cols3].copy()

merged = pd.merge(
    new_df1, 
    new_df2, 
    on='TRAVELER_ID', 
    how='inner'
).merge(
    new_df3, 
    on='TRAVELER_ID', 
    how='inner'
)
merged = merged.drop_duplicates()
merged.sort_values('TRAVELER_ID', inplace=True)
merged.to_csv("preprocessed_travel_data.csv", index=False)
