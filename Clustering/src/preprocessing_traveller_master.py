import pandas as pd
import numpy as np

region = "F"

df = pd.read_csv(f"data/raw_data/{region}/tn_traveller_master_여행객 Master_{region}.csv")

df['GENDER'] = df['GENDER'].map({"남": 0, "여": 1})
df['AGE_GRP'] = df['AGE_GRP'].map({20: 0, 30: 0.25, 40: 0.5, 50: 0.75, 60: 1})
df = df[df['MARR_STTS'] != 5] #'기타'항목 (결측치) 인 행 삭제
df['MARR_STTS'] = df['MARR_STTS'].apply(lambda x: 1 if x == 2 else 0) # 1: 기혼(2), 0: 미혼(2 외)
df = df[df['JOB_NM'] != 13] #'기타'항목 (결측치) 인 행 삭제
df['JOB_NM'] = df['JOB_NM'].apply(lambda x: 1 if x == 12 else 0) # 1: 학생(12), 0: 직장인 혹은 전업주부(12 외)
df['INCOME'] = (df['INCOME'] - 1) / (12 - 1)
df = df[df['HOUSE_INCOME'].notna()] # na (결측치) 인 행 삭제
df['HOUSE_INCOME'] = (df['HOUSE_INCOME'] - 1) / (12 - 1)

cols = [f"TRAVEL_STYL_{i}" for i in range(1, 9)]
for col in cols:
    df[col] = (df[col] - 1) / 6

# 연간 여행 빈도 계산
#1. 기타 (결측치) -> 평균으로 처리
cond_1 = df['TRAVEL_TERM'] == 1 # 1주일기준
cond_2 = df['TRAVEL_TERM'] == 2 # 월기준
cond_3 = df['TRAVEL_TERM'] == 3 # 연기준
cond_4 = df['TRAVEL_TERM'] == 4 # 기타

values = [
    df['TRAVEL_NUM'] * 52,
    df['TRAVEL_NUM'] * 12,
    df['TRAVEL_NUM'] * 1,
    np.nan  # 일단 4인 경우는 NaN으로 처리
]

df['TRAVEL_FREQ'] = np.select([cond_1, cond_2, cond_3, cond_4], values)

mean_freq = df.loc[df['TRAVEL_TERM'] != 4, 'TRAVEL_FREQ'].mean() # 4 제외한 값들로 평균
df['TRAVEL_FREQ'] = df['TRAVEL_FREQ'].fillna(mean_freq) # 4인 경우 (NaN인 경우) 평균으로 채우기

#2. 이상치 제거
Q1 = df['TRAVEL_FREQ'].quantile(0.25)
Q3 = df['TRAVEL_FREQ'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['TRAVEL_FREQ'] >= lower_bound) & (df['TRAVEL_FREQ'] <= upper_bound)]

#3. 정규화
min_val = df['TRAVEL_FREQ'].min()
max_val = df['TRAVEL_FREQ'].max()
df['TRAVEL_FREQ'] = (df['TRAVEL_FREQ'] - min_val) / (max_val - min_val)

    
df = df[['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'MARR_STTS', 'JOB_NM', 'INCOME', 'HOUSE_INCOME', 'TRAVEL_FREQ', 'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8']]

"""
print(df.head())         # 앞 5개 행
print(df.info())         # 열 정보 및 결측치 확인
print(df.describe())     # 수치형 요약 통계
"""

df.to_csv(f"data/preprocessed_data/{region}/preprocessed_traveller_master_{region}.csv", index=False)

print("전처리완료")