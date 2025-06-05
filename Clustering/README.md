## 구현 방식
1차 분류: 성별, 연령대를 기준으로 그룹화
2차 분류: 각 그룹 내에서 여행 스타일 설문 항목을 6개의 값을 기반으로 K-Means 클러스터링 기법을 활용하여 3개의 하위 클러스터로 분류

## 분석 기법 선정 이유
K-Means는 구현이 간단하고 직관적인 클러스터링 기법이기 때문에 선택
클러스터 수를 3개로 했을 때 PCA 시각화 결과가 가장 잘 구분되어 보여 해당 수로 결정

## 예상 vs 결과
예상: 성별, 연령대에 따라 선호하는 여행 스타일이 다를 것이라고 예상, 특정 여행 스타일 문항이 모든 성별, 연령대 그룹에서 공통적으로 클러스터 구분에 중요하게 작용할 것이라고 예상함
결과: F값, p값 분석 결과 전반적으로 대부분의 그룹에서 클러스터 간 유의미한 차이(p < 0.05) 가 나타났음. 예상과 다르게 클러스터링에 영향력이 큰 변수는 그룹마다 달랐음.

=== GENDER=0, AGE_GRP=0.0 ===
TRAVEL_STYL_7: F = 1025.07, p = 0.0000
TRAVEL_STYL_1: F = 303.40, p = 0.0000
TRAVEL_STYL_6: F = 235.95, p = 0.0000
TRAVEL_STYL_3: F = 211.89, p = 0.0000
TRAVEL_STYL_8: F = 15.77, p = 0.0000
TRAVEL_STYL_5: F = 2.08, p = 0.1259

=== GENDER=0, AGE_GRP=0.25 ===
TRAVEL_STYL_7: F = 1872.74, p = 0.0000
TRAVEL_STYL_6: F = 416.15, p = 0.0000
TRAVEL_STYL_1: F = 119.03, p = 0.0000
TRAVEL_STYL_3: F = 50.62, p = 0.0000
TRAVEL_STYL_5: F = 46.17, p = 0.0000
TRAVEL_STYL_8: F = 23.44, p = 0.0000

=== GENDER=0, AGE_GRP=0.5 ===
TRAVEL_STYL_5: F = 344.67, p = 0.0000
TRAVEL_STYL_3: F = 197.46, p = 0.0000
TRAVEL_STYL_1: F = 164.94, p = 0.0000
TRAVEL_STYL_7: F = 143.51, p = 0.0000
TRAVEL_STYL_6: F = 73.48, p = 0.0000
TRAVEL_STYL_8: F = 67.17, p = 0.0000

=== GENDER=0, AGE_GRP=0.75 ===
TRAVEL_STYL_5: F = 271.96, p = 0.0000
TRAVEL_STYL_3: F = 161.26, p = 0.0000
TRAVEL_STYL_6: F = 138.09, p = 0.0000
TRAVEL_STYL_1: F = 47.50, p = 0.0000
TRAVEL_STYL_7: F = 31.15, p = 0.0000
TRAVEL_STYL_8: F = 23.59, p = 0.0000

=== GENDER=0, AGE_GRP=1.0 ===
TRAVEL_STYL_5: F = 112.87, p = 0.0000
TRAVEL_STYL_7: F = 72.06, p = 0.0000
TRAVEL_STYL_6: F = 58.99, p = 0.0000
TRAVEL_STYL_3: F = 21.85, p = 0.0000
TRAVEL_STYL_1: F = 16.08, p = 0.0000
TRAVEL_STYL_8: F = 6.99, p = 0.0012

=== GENDER=1, AGE_GRP=0.0 ===
TRAVEL_STYL_7: F = 4475.87, p = 0.0000
TRAVEL_STYL_5: F = 639.03, p = 0.0000
TRAVEL_STYL_8: F = 103.31, p = 0.0000
TRAVEL_STYL_6: F = 20.12, p = 0.0000
TRAVEL_STYL_3: F = 16.60, p = 0.0000
TRAVEL_STYL_1: F = 12.12, p = 0.0000

=== GENDER=1, AGE_GRP=0.25 ===
TRAVEL_STYL_7: F = 1884.27, p = 0.0000
TRAVEL_STYL_3: F = 364.50, p = 0.0000
TRAVEL_STYL_1: F = 319.36, p = 0.0000
TRAVEL_STYL_6: F = 210.57, p = 0.0000
TRAVEL_STYL_8: F = 27.14, p = 0.0000
TRAVEL_STYL_5: F = 1.26, p = 0.2831

=== GENDER=1, AGE_GRP=0.5 ===
TRAVEL_STYL_5: F = 598.97, p = 0.0000
TRAVEL_STYL_3: F = 253.31, p = 0.0000
TRAVEL_STYL_7: F = 142.60, p = 0.0000
TRAVEL_STYL_8: F = 123.62, p = 0.0000
TRAVEL_STYL_6: F = 99.91, p = 0.0000
TRAVEL_STYL_1: F = 72.56, p = 0.0000

=== GENDER=1, AGE_GRP=0.75 ===
TRAVEL_STYL_8: F = 336.90, p = 0.0000
TRAVEL_STYL_3: F = 96.77, p = 0.0000
TRAVEL_STYL_6: F = 92.38, p = 0.0000
TRAVEL_STYL_7: F = 83.23, p = 0.0000
TRAVEL_STYL_1: F = 50.45, p = 0.0000
TRAVEL_STYL_5: F = 20.73, p = 0.0000

=== GENDER=1, AGE_GRP=1.0 ===
TRAVEL_STYL_1: F = 90.84, p = 0.0000
TRAVEL_STYL_3: F = 88.29, p = 0.0000
TRAVEL_STYL_5: F = 68.25, p = 0.0000
TRAVEL_STYL_7: F = 59.19, p = 0.0000
TRAVEL_STYL_6: F = 26.80, p = 0.0000
TRAVEL_STYL_8: F = 17.04, p = 0.0000

## 의의
비슷한 여행 성향을 가진 사람들끼리 군집화
예상대로 성별 및 연령대에 따라 중요한 여행 스타일 요인이 달라졌음
이는 클러스터링이 단순히 한두 변수에 의존하지 않고 복합적인 요인에 의해 형성되었음을 시사
클러스터 수를 3개로 했던 판단도 적절했으며, 분산 분석 결과가 이를 뒷받침


## 할 것
실루엣 코이피션트 기준으로 최적의 클러스터 개수 다시 계산(원래 이렇게 했어야 하는데 까먹었어요 죄송합니다..)
DBSCAN, 계층적 클러스터링 등 다른 방법으로 클러스터링 해보고 비교
각 클러스터 별 특징 분석, 클러스터간 비교




[0602 데이터 전처리 완료]

-전처리 된 csv 파일 위치: Clustering/0602/data/preprocessed_data에 권역별로 있습니다.<br />
preprocessed_with_cluster_{region}.csv는 전처리된 preprocessed_{region}.csv에 clustered_k=3_per_gender&age.csv를 기준으로 클러스터링 결과를 합친 것입니다('CLUSTER' 컬럼 추가됨. 각 성별, 연령대 별로 0.0, 1.0, 2.0 세개의 클러스터 존재).

-전처리 내용: tn_traveller_master_여행객 Master_{region}.csv, tn_travel_여행_{region}.csv, tn_visit_area_info_방문지정보_{region}.csv, tn_companion_info_동반자정보_{region}.csv 테이블에서 필요한 컬럼만 남긴 뒤 조인했습니다. 구체적인 컬럼들은 아래와 같습니다.<br />

여행객 테이블: ['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8', 'TRAVEL_STATUS_ACCOMPANY'] (여행스타일 2,4는 숙박 관련 사항이라 제외)<br />
여행 테이블: ['TRAVEL_ID', 'TRAVELER_ID', 'TRAVEL_START_YMD', 'TRAVEL_END_YMD', 'MVMN_NM']<br />
방문지정보 테이블: ['TRAVEL_ID', 'VISIT_AREA_NM', 'VISIT_AREA_TYPE_CD', 'DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']<br />
동반자정보 테이블: ['TRAVEL_ID', 'REL_CD']<br />

-세부사항<br />
.총 여행일수를 나타내는 'TRAVEL_DAYS' 컬럼, 여행월을 나타내는 'MAIN_TRAVEL_MONTH'(두달에 걸칠 경우 더 많은 날이 포함된 월 기준) 컬럼 추가<br />
.'VISIT_AREA_TYPE_CD'이 1, 2, 3, 4, 5, 6, 7, 8, 12, 13인 것만 포함 (집, 역, 식당 등 제외, 기타 포함)<br />
.각 여행 별 동반자 타입을 나타내는 'RELATION_TYPE' 컬럼 추가 ('가족', '친구', '기타', '혼자' 중 1)<br />
.각 여행객들의 평균 'DGSTFN', 평균 'REVISIT_INTENTION', 평균 'RCMDTN_INTENTION'을 구해 그 값이 하나라도 [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]를 벗어나면 아웃라이어로 보고 제거<br />
.자세한 것은 preprocessing.py 참고 바람

[0602 데이터 클러스터링 완료]

-클러스터링 된 csv 파일 위치: Clustering/0602/data/clustered_data에 있습니다.

-'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8'을 이용해 클러스터링했습니다.

-세부사항<br />
1. clustered_k=n_per_gender&age.csv: E,F,G,H의 여행객들 데이터를 모두 통합하여 클러스터링 진행. 각 {성별, 나이대} 별로 여행스타일에 따라 n개의 클러스터로 나눔(따라서 총 10*n개의 클러스터). PCA로 2차원에 그려본 결과 k=3일 때 가장 시각적으로 잘 분리됨(동일한 이름의 .png 파일 참고 바람) <br />
2. clustered_k=n.csv: E,F,G,H의 여행객들 데이터를 모두 통합하여 클러스터링 진행. 성별, 나이대도 함께 변수로 포함하여 클러스터링 진행. PCA로 2차원에 그려본 결과 k=8일 때 가장 시각적으로 잘 분리되나(동일한 이름의 .png 파일 참고 바람) 성별, 나이에 따라 분리된 것으로 보임<br />

-clustered_k=3_per_gender&age.csv이 가장 적절해 보임. 추후 각 클러스터별 여행 성향 분석해서 업로드 하겠습니다.
