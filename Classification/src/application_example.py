# 장소 리스트
places = [
    "경기전", "금능해수욕장", "비자림", "서귀포 매일 올레시장", "성산일출봉",
    "속초관광수산시장", "오설록 티 뮤지엄", "전주한옥마을", "정방폭포", "제주동문시장",
    "제주동문재래 야시장", "제주동문재래시장", "천지연폭포", "초원 사진관", "혼인지"
]

# 사용자 공통 조건 (22번 클러스터 기반)
common_info = {
    "MAIN_TRAVEL_MONTH": 8,
    "TRAVEL_STATUS_ACCOMPANY": "2인 여행(가족 외)",
    "RELATION_TYPE": "친구",
    "MVMN_NM": "대중교통 등",
    "GENDER": 1,
    "AGE_GRP": 0.75,
    "VISIT_AREA_TYPE_CD": 1,
    "TRAVEL_STYL_1": 0.7,
    "TRAVEL_STYL_3": 0.4,
    "TRAVEL_STYL_5": 0.8,
    "TRAVEL_STYL_6": 0.2,
    "TRAVEL_STYL_7": 0.9,
    "TRAVEL_STYL_8": 0.6
}

# 입력 데이터 생성
new_data = pd.DataFrame([{**common_info, "VISIT_AREA_NM": place} for place in places])

# 예측 수행
pred_labels = pipeline.predict(new_data)
pred_proba = pipeline.predict_proba(new_data)

# 결과 정리
label_map = {0: "Dissatisfied", 1: "Neutral", 2: "Satisfied"}
results = []
for i, place in enumerate(places):
    results.append({
        "VISIT_AREA_NM": place,
        "Predicted_Label": label_map[pred_labels[i]],
        "Satisfied_Prob": round(float(pred_proba[i][2]), 2),
        "Neutral_Prob": round(float(pred_proba[i][1]), 2),
        "Dissatisfied_Prob": round(float(pred_proba[i][0]), 2)
    })

df_results = pd.DataFrame(results)

# 만족 확률 기준 정렬
df_results_sorted = df_results.sort_values(by="Satisfied_Prob", ascending=False).reset_index(drop=True)
print(df_results_sorted)
