import pandas as pd
import numpy as np

region = 'H' #지역설정

#원본 파일 불러오기
df1 = pd.read_csv(f'Clustering/0602/data/original_data/{region}/tn_traveller_master_여행객 Master_{region}.csv')
df2 = pd.read_csv(f'Clustering/0602/data/original_data/{region}/tn_travel_여행_{region}.csv')
df3 = pd.read_csv(f'Clustering/0602/data/original_data/{region}/tn_visit_area_info_방문지정보_{region}.csv')
df4 = pd.read_csv(f'Clustering/0602/data/original_data/{region}/tn_companion_info_동반자정보_{region}.csv')

#쓸 변수(컬럼)만 뽑기
df1 = df1[['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8', 'TRAVEL_STATUS_ACCOMPANY']]
df2 = df2[['TRAVEL_ID', 'TRAVELER_ID', 'TRAVEL_START_YMD', 'TRAVEL_END_YMD', 'MVMN_NM']]
df3 = df3[['TRAVEL_ID', 'VISIT_AREA_NM', 'VISIT_AREA_TYPE_CD', 'DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']]
df4 = df4[['TRAVEL_ID', 'REL_CD']]

#변수값 전처리
df1['GENDER'] = df1['GENDER'].map({"남": 0, "여": 1})
df1['AGE_GRP'] = df1['AGE_GRP'].map({20: 0, 30: 0.25, 40: 0.5, 50: 0.75, 60: 1})
df1['TRAVEL_STYL_1'] = (df1['TRAVEL_STYL_1'] - 1) / 6
df1['TRAVEL_STYL_3'] = (df1['TRAVEL_STYL_3'] - 1) / 6
df1['TRAVEL_STYL_5'] = (df1['TRAVEL_STYL_5'] - 1) / 6
df1['TRAVEL_STYL_6'] = (df1['TRAVEL_STYL_6'] - 1) / 6
df1['TRAVEL_STYL_7'] = (df1['TRAVEL_STYL_7'] - 1) / 6
df1['TRAVEL_STYL_8'] = (df1['TRAVEL_STYL_8'] - 1) / 6

df2['TRAVEL_START_YMD'] = pd.to_datetime(df2['TRAVEL_START_YMD'])
df2['TRAVEL_END_YMD'] = pd.to_datetime(df2['TRAVEL_END_YMD'])
 #여행일수 변수 추가 
df2['TRAVEL_DAYS'] = (df2['TRAVEL_END_YMD'] - df2['TRAVEL_START_YMD']).dt.days + 1 
 #여행시기 변수 추가 (1~12의 정수)
def get_main_travel_month(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    month_counts = date_range.month.value_counts()
    return month_counts.idxmax()
df2['MAIN_TRAVEL_MONTH'] = df2.apply(
    lambda row: get_main_travel_month(row['TRAVEL_START_YMD'], row['TRAVEL_END_YMD']), axis=1
)

df3 = df3[df3['VISIT_AREA_TYPE_CD'].isin([1, 2, 3, 4, 5, 6, 7, 8, 12, 13])] #기타포함

def categorize_relation(code):
    if 1 <= code <= 6:
        return '가족'
    elif 7 <= code <= 10:
        return '친구'
    elif code == 11:
        return '기타' #기타포함
    else:
        return '기타'  
    #left outer join 시 동반인이 없을 경우 NaN값 --> '혼자'로 수정 예정
df4['RELATION_TYPE'] = df4['REL_CD'].apply(categorize_relation)

#join연산
df_12 = pd.merge(df1, df2, on='TRAVELER_ID', how='inner')
df_123 = pd.merge(df_12, df3, on='TRAVEL_ID', how='inner')
df_1234= pd.merge(df_123, df4, on='TRAVEL_ID', how='left') #동반자 여러명일 경우 중복행 발생(*)
df_1234['RELATION_TYPE'] = df_1234['RELATION_TYPE'].fillna('혼자')
df_1234 = df_1234.drop(columns=['REL_CD'])
df_1234.drop_duplicates(inplace=True) #중복행 제거(*)



#아웃라이어 제거
#IQR 방식 이용, 'DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION' 중 하나라도 평균에서 일정 범위를 벗어나면 제거
traveler_avg = df_1234.groupby('TRAVELER_ID')[['DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']].mean().reset_index()
def get_outlier_flags(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return ~series.between(lower, upper)
outlier_dgstfn = get_outlier_flags(traveler_avg['DGSTFN'])
outlier_revisit = get_outlier_flags(traveler_avg['REVISIT_INTENTION'])
outlier_rcmdtn = get_outlier_flags(traveler_avg['RCMDTN_INTENTION'])
traveler_avg['is_outlier'] = outlier_dgstfn | outlier_revisit | outlier_rcmdtn
valid_ids = traveler_avg.loc[~traveler_avg['is_outlier'], 'TRAVELER_ID']
df_filtered = df_1234[df_1234['TRAVELER_ID'].isin(valid_ids)].copy()

#제거된 traveler 개수 확인용 (E: 2509 2475 34, F: 2521 2490 31, G: 2531 2493 38, H:2530 2486 44)
"""
original_count = df_1234['TRAVELER_ID'].nunique()
filtered_count = df_filtered['TRAVELER_ID'].nunique()
removed_count = original_count - filtered_count
print(original_count, filtered_count, removed_count)

"""

#저장
df_filtered.to_csv(f"Clustering/0602/data/preprocessed_data/{region}/preprocessed_{region}.csv", index=False)
print("전처리완료")