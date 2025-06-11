
## ✅ 최종 채택된 모델

- Random Forest + SMOTE + class_weight 조합 사용함  
- 분류 기준:  
  - DGSTFN 1~2점 → 불만족 (0)  
  - DGSTFN 3~4점 → 보통 (1)  
  - DGSTFN 5점 → 만족 (2)

---

## 🧪 성능 요약 (전체 테스트셋 기준)

| 지표 | 값 |
|------|-----|
| Accuracy | 63.8% |
| Macro F1-score | 0.46 |
| 클래스별 F1 | 불만족: 0.09 / 보통: 0.63 / 만족: 0.66 |

→ 관련 수치는 `classification_report.csv` 및 `confusion_matrix.csv`에 정리해둠

---

## 🔍 클러스터별 개별 모델 vs 통합 모델 비교

초기에는 GENDER, AGE_GRP 기준으로 데이터셋을 나누고  
클러스터별로 개별 모델을 학습하는 전략도 실험했음.  

그러나 성능 비교 결과, **모든 클러스터에서 통합 모델이 Macro F1 기준 더 우수**함을 확인함.  
따라서 개별 모델 대신 **통합 모델 하나만 사용하는 방식으로 최종 결정함**.

아래는 Macro F1 기준으로 클러스터별 성능을 비교한 시각화임:

![클러스터별_모델비교_MacroF1](./클러스터별_모델비교_MacroF1.png)

---

## 💻 주요 파일 설명

- `final_model_pipeline.py`  
 → 전체 전처리 및 통합 모델 학습 파이프라인 구현함
- `통합모델_classification_report.csv`  
 → 클래스별 precision / recall / f1-score 결과 저장함
- `통합모델_confusion_matrix.csv`  
 → 예측 vs 실제 값의 분포 정리한 confusion matrix
- `클러스터별_모델비교_MacroF1.png`  
 → 통합 vs 개별 모델 성능 비교 시각화
