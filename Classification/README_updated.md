
# 🧭 Classification - Satisfaction Prediction

본 폴더는 여행 로그 데이터를 기반으로 사용자 만족도를 예측하는 classification 모델 실험 결과를 정리한 것입니다.

---

## 📁 파일 설명

### 📂 data/
- `preprocessed_with_cluster_ALL.csv`: 전처리 및 클러스터링이 포함된 통합 데이터셋 (모델 학습용)

### 📂 result/
- `clusterwise_group_rf_results.csv`: 성별/연령/클러스터 그룹별 회귀 성능 지표 (MAE, R² 등)
- `feature_importance_rf.png`: RandomForest 기반 변수 중요도 시각화 결과

---

## 🧠 모델 설계 및 전처리

- 타겟 변수는 `DGSTFN` (만족도)이며,  
  분류 모델에서는 `DGSTFN ≥ 4.0` → 1, 그 외는 0으로 이진화하여 사용함
- 베이스라인 분류 모델은 **로지스틱 회귀**,  
  본 모델은 **랜덤포레스트 다중회귀(RandomForestRegressor)** 적용

### 🔹 입력 변수 구성

| 변수 | 처리 방식 |
|------|-----------|
| `TRAVEL_STYL_1`, ..., `TRAVEL_STYL_8` | 수치형 그대로 사용 |
| `VISIT_AREA_NM`, `MAIN_TRAVEL_MONTH` | OneHot 인코딩 |
| `TRAVEL_STATUS_ACCOMPANY`, `RELATION_TYPE`, `MVMN_NM` | OneHot 인코딩 |
| `GENDER`, `AGE_GRP`, `VISIT_AREA_TYPE_CD` | OneHot 인코딩 |
| `TRAVEL_ID`, `TRAVELER_ID`, `TRAVEL_START_YMD`, `TRAVEL_END_YMD`, `REVISIT_INTENTION`, `RCMDTN_INTENTION` | 예측과 무관하여 제거 |
| `CLUSTER` | 전체 모델에서는 제외, 클러스터별 모델에서는 사용 |

- 클러스터링은 성별(GENDER)과 연령대(AGE_GRP)를 기준으로 k=3 clustering
- 전체 모델 외에도 각 성별+연령 그룹 내에서 **클러스터별 회귀 모델**을 별도 학습하여 비교

---

## 📈 모델 성능 요약

### 🔹 분류 (Logistic Regression)
- Accuracy: **70.73%**
- Class 0 F1-score: **0.26**
- Class 1 F1-score: **0.82**

### 🔹 회귀 (Random Forest)
- 전체 모델 R²: **0.6254**
- MAE: **0.2601**
- 클러스터별 평균 R²: **0.6051**
- 최고 그룹 R²: **0.7661**, 최저: **0.2766**

---

## 📌 핵심 인사이트
- `RCMDTN_INTENTION`, `REVISIT_INTENTION` 은 예측 성능에 영향력이 크지만, 결과 변수이므로 모델 입력에서는 제거됨
- 성별/연령에 따른 그룹별 회귀 분석 시, 일부 남성 그룹에서 설명력 매우 높음
