
## 📄 1. `update_logisticregression.py`

- **모델 유형:** 이진 분류 (Logistic Regression)
- **타겟 정의:** DGSTFN > 4.0 → 만족 (1), 그렇지 않으면 불만족 (0)
- **전처리:** 범주형 변수에 OneHotEncoding 적용, TRAVEL_STYLE 변수는 수치형 그대로 사용
- **클래스 불균형 대응:** `class_weight="balanced"` 옵션 적용
- **주요 시도**
  - 원래 기준인 `DGSTFN >= 4.0` 대신 `> 4.0`을 기준으로 변경
  - 만족도를 좀 더 보수적으로 정의함으로써 class 0과 1의 비율이 균형에 가까워지도록 조정
- **결과 요약:** 클래스 간 균형은 맞춰졌으나 전체 정확도는 감소하여 예측 모델의 실효성에는 의문이 남음

---

## 📄 2. `classification_rf_freq.py`

- **모델 유형:** 회귀(Random Forest Regressor) → 예측값을 반올림하여 정수형 만족도로 해석
- **타겟 변수:** DGSTFN (정수형: 1~5)
- **주요 시도**
  - 예측 결과가 1~5 정수 값에 더 가까워지도록 `np.round()`로 반올림 후 `clip` 처리
  - `VISIT_AREA_NM`, `VISIT_AREA_TYPE_CD` 변수에는 **Frequency Encoding** 적용  
    → 희소한 관광지로 인한 학습 왜곡을 완화하기 위함
  - 나머지 범주형 변수는 OneHotEncoding 사용
- **결과 요약:** 정수형 만족도 예측이 가능해졌으나, 여전히 예측값이 3에 몰리는 경향 있음. 일부 중요한 변수의 편향 영향 가능성 제기

---

## 📄 3. `update_xgboost_visualize.py`

- **모델 유형:** 회귀 (XGBoost)
- **타겟 변수:** DGSTFN (연속형 만족도 점수 예측)
- **주요 시도**
  - `VISIT_AREA_NM`, `VISIT_AREA_TYPE_CD`에 Frequency Encoding 적용
  - 나머지 범주형 변수에 OneHotEncoding 적용
  - TRAVEL_STYLE 변수는 수치형 그대로 입력
  - 모델 결과를 시각화하여 예측값이 왜 특정 점수에 집중되는지 분석 (`예: 히스토그램 시각화`)
- **결과 요약:** 성능은 소폭 개선되었으나 만족도 3 근처에 예측값이 몰리는 현상은 여전히 존재

---

## 📌 공통 고려 사항

- `REVISIT_INTENTION`, `RCMDTN_INTENTION`은 모델 학습에서 제외함 (미래 예측에 사용 불가한 변수)
- TRAVEL_STYLE 관련 변수는 OneHotEncoding 없이 수치형 그대로 사용
- 데이터 불균형, 특정 만족도(5점)의 과도한 집중 등 구조적 문제로 인해 예측 한계 존재
