## 최종 모델 & 데이터
최종 모델: Clustering/0602/src/final_K-means_(1).py<br />
최종 데이터: Clustering/0602/data/clustered_data/clustered_k=3_per_gender&age.csv

## 구현 방식
1차 분류: 성별, 연령대를 기준으로 그룹화<br />
2차 분류: 각 그룹 내에서 여행 스타일 설문 항목을 6개의 값을 기반으로 K-Means 클러스터링 기법을 활용하여 3개의 하위 클러스터로 분류

## 결과 평가
Silhouette Coefficient가 대략 0.16 정도로 뚜렷하게 나뉘지 않고 약한 경향성이 나타나는 것으로 보임

<br /><br />
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
