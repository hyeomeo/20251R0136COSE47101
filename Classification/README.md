
# 📊 Classification: Satisfaction Prediction Model

## 📁 Folder Structure

```
classification/
├── data/
│   ├── preprocessed_with_cluster_ALL.csv        # 전처리 및 클러스터링 통합 데이터
│   └── clusterwise_group_rf_results.csv         # 클러스터별 회귀 결과 저장
│
├── results/
│   ├── feature_importance_rf.png                # 변수 중요도 시각화 결과
│
├── notebooks/
│   └── classification_baseline.ipynb            # Colab 실험용 노트북
│
├── src/
│   └── satisfaction_model_final.py              # 전체 분석 코드 (.py 스크립트 버전)
```

## 🧠 Project Overview

- 여행 로그 데이터를 기반으로 사용자 만족도(DGSTFN)를 예측하는 모델을 개발함
- 분류 (DGSTFN >= 4.0 → 1, else 0) 및 회귀 (DGSTFN 점수 자체)를 모두 수행
- 성별(GENDER), 연령대(AGE_GRP), 여행스타일, 방문지, 의도변수 등을 활용
- 클러스터링 정보는 그룹 내 모델에만 포함하여 성능 비교 수행

## ✅ Overall Model Results

### Classification (Logistic Regression)
- **Accuracy:** 87.75%
- **Class 0 F1-score:** 0.66
- **Class 1 F1-score:** 0.93

### Regression (Random Forest)
- **MAE:** 0.2601
- **MSE:** 0.2433
- **R²:** 0.6254

### Variable Importance
- `RCMDTN_INTENTION`, `REVISIT_INTENTION` 이 가장 중요한 변수로 나타남
- `TRAVEL_STYL_*`, `AGE_GRP`, `MAIN_TRAVEL_MONTH` 등도 일부 기여

![Feature Importance](results/feature_importance_rf.png)

### Cluster-wise Regression Results
- 클러스터별 회귀 모델의 평균 R²: **0.6051**
- 최고 R²: **0.7661**, 최저 R²: **0.2766**
- 성별/연령대에 따라 설명력 차이가 있으며, 일부 남성 그룹에서 성능 우수

➡️ 전체 결과는 `data/clusterwise_group_rf_results.csv` 참조
