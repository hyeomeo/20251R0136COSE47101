[0602 데이터 전처리 완료]

-전처리 된 csv 파일 위치: Clustering/0602/data/preprocessed_data에 권역별로 있습니다.

-전처리 내용: tn_traveller_master_여행객 Master_{region}.csv, tn_travel_여행_{region}.csv, tn_visit_area_info_방문지정보_{region}.csv, tn_companion_info_동반자정보_{region}.csv 테이블에서 필요한 컬럼만 남긴 뒤 조인했습니다. 구체적인 컬럼들은 아래와 같습니다.

여행객 테이블: ['TRAVELER_ID', 'GENDER', 'AGE_GRP', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8', 'TRAVEL_STATUS_ACCOMPANY'] (여행스타일 2,4는 숙박 관련 사항이라 제외)
여행 테이블: ['TRAVEL_ID', 'TRAVELER_ID', 'TRAVEL_START_YMD', 'TRAVEL_END_YMD', 'MVMN_NM']
방문지정보 테이블: ['TRAVEL_ID', 'VISIT_AREA_NM', 'VISIT_AREA_TYPE_CD', 'DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']
동반자정보 테이블: ['TRAVEL_ID', 'REL_CD']

-세부사항
!총 여행일수를 나타내는 'TRAVEL_DAYS' 컬럼, 여행월을 나타내는 'MAIN_TRAVEL_MONTH'(두달에 걸칠 경우 더 많은 날이 포함된 월 기준) 컬럼 추가
!'VISIT_AREA_TYPE_CD'이 1, 2, 3, 4, 5, 6, 7, 8, 12, 13인 것만 포함 (집, 역, 식당 등 제외, 기타 포함)
!각 여행 별 동반자 타입을 나타내는 'RELATION_TYPE' 컬럼 추가 ('가족', '친구', '기타', '혼자' 중 1)
!각 여행객들의 평균 'DGSTFN', 평균 'REVISIT_INTENTION', 평균 'RCMDTN_INTENTION'을 구해 그 값이 하나라도 [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]를 벗어나면 아웃라이어로 보고 제거
!자세한 것은 preprocessing.py 참고 바람

