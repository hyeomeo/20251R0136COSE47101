import pandas as pd
import numpy as np

region = 'H' #지역설정

df1 = pd.read_csv(f'data/raw_data/{region}/tn_traveller_master_여행객 Master_{region}.csv')
df2 = pd.read_csv(f'data/raw_data/{region}/tn_travel_여행_{region}.csv')
df3 = pd.read_csv(f'data/raw_data/{region}/tn_visit_area_info_방문지정보_{region}.csv')

df1 = df1[['TRAVELER_ID']]
df2 = df2[['TRAVEL_ID', 'TRAVELER_ID']]
df3 = df3[['TRAVEL_ID', 'VISIT_AREA_TYPE_CD', 'DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']]

df_merged = df2.merge(df3, on='TRAVEL_ID') # inner join

df_filtered = df_merged[df_merged['VISIT_AREA_TYPE_CD'].isin(range(1, 14))] # 이외의 VIS 값은 집, 숙소, 사무실 등 여행지에 해당 X 이므로 필터링
traveler_trip_counts = df2.groupby('TRAVELER_ID')['TRAVEL_ID'].nunique().rename("TOTAL_TRIPS") # 각 TRAVELER_ID별 총 여행 수 계산

agg_df = df_filtered.groupby(['TRAVELER_ID', 'VISIT_AREA_TYPE_CD']).agg(  
    VISIT_COUNT=('VISIT_AREA_TYPE_CD', 'count'), # 해당 방문지 유형 총 방문 횟수 (전여행 통틀어)
    DGSTFN_SUM=('DGSTFN', 'sum'), # 해당 방문지 유형 누적 만족도
    DGSTFN_COUNT=('DGSTFN', 'count') # 만족도 결측치는 개수에서 제외
).reset_index()

agg_df = agg_df.merge(traveler_trip_counts, on='TRAVELER_ID')
agg_df['AVG_VISIT_PER_TRIP'] = agg_df['VISIT_COUNT'] / agg_df['TOTAL_TRIPS'] # 여행객, 여행지유형 별 여행당 평균 방문 횟수
agg_df['AVG_DGSTFN'] = agg_df['DGSTFN_SUM'] / agg_df['DGSTFN_COUNT'] # 여행객, 여행지유형 별 평균 만족도
global_avg_dgstfn = agg_df.groupby('VISIT_AREA_TYPE_CD')['AVG_DGSTFN'].mean().rename("GLOBAL_AVG_DGSTFN") # 여행지유형 별 여행객들의 평균 만족도 (특정 유형 방문정보 결측 시 평균으로 채워넣기 위해)

traveler_ids = df1['TRAVELER_ID'].unique()
visit_types = list(range(1, 14)) 
full_index = pd.MultiIndex.from_product([traveler_ids, visit_types], names=["TRAVELER_ID", "VISIT_AREA_TYPE_CD"]) # VISIT_AREA_TYPE_CD 1~13 모두 포함하는 모든 조합 생성 (방문정보 결측 시 na 값)
full_df = pd.DataFrame(index=full_index).reset_index()

result = full_df.merge(agg_df[['TRAVELER_ID', 'VISIT_AREA_TYPE_CD', 'AVG_VISIT_PER_TRIP', 'AVG_DGSTFN']], 
                       on=['TRAVELER_ID', 'VISIT_AREA_TYPE_CD'], how='left')

result['AVG_VISIT_PER_TRIP'] = result['AVG_VISIT_PER_TRIP'].fillna(0) # 특정 유형의 방문지 방문 정보 결측시 방문 빈도는 0으로, 
result = result.merge(global_avg_dgstfn, on='VISIT_AREA_TYPE_CD', how='left')
result['AVG_DGSTFN'] = result['AVG_DGSTFN'].fillna(result['GLOBAL_AVG_DGSTFN']) # 만족도는 전역 평균으로 채움
result = result.drop(columns='GLOBAL_AVG_DGSTFN')

visit_pivot = result.pivot(index='TRAVELER_ID', columns='VISIT_AREA_TYPE_CD', values='AVG_VISIT_PER_TRIP')
visit_pivot.columns = [f'AVG_VISIT_PER_TRIP_TYPE_{col}' for col in visit_pivot.columns]

dgstfn_pivot = result.pivot(index='TRAVELER_ID', columns='VISIT_AREA_TYPE_CD', values='AVG_DGSTFN')
dgstfn_pivot.columns = [f'AVG_DGSTFN_TYPE_{col}' for col in dgstfn_pivot.columns]

final_df = pd.concat([visit_pivot, dgstfn_pivot], axis=1).reset_index()

#print(final_df)


# 1. 이상치 제거 (IQR 기준)
def remove_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = max(Q3 - Q1, 1) # Q3 == Q1인 경우 방지
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.where((series >= lower_bound) & (series <= upper_bound), np.nan) # 이상치면 NaN으로 처리

# 2. 정규화 함수 (Min-Max)
def min_max_scale(series):
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val: # 값이 모두 같으면 0으로 변환 (분모 0 방지)
        return series*0
    return (series - min_val) / (max_val - min_val)

visit_cols = [f'AVG_VISIT_PER_TRIP_TYPE_{i}' for i in range(1,14)]
for col in visit_cols:
    final_df[col] = remove_outliers_iqr(final_df[col])
    final_df[col] = min_max_scale(final_df[col])

dgstfn_cols = [f'AVG_DGSTFN_TYPE_{i}' for i in range(1,14)]

for col in dgstfn_cols:
    final_df[col] = (final_df[col] - 1) / (5 - 1)

final_df = final_df.dropna()

final_df.to_csv(f"data/preprocessed_data/{region}/preprocessed_visit_area_info_{region}.csv", index=False)

print("전처리완료")