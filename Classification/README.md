## 최종 채택 모델

- Random Forest + SMOTE + class_weight
- 분류 기준:  
  - DGSTFN 1~2점 → 불만족 (0)  
  - DGSTFN 3~4점 → 보통 (1)  
  - DGSTFN 5점 → 만족 (2)

---

## 성능 요약 (전체 테스트셋 기준)

| 지표 | 값 |
|------|-----|
| Accuracy | 63.8% |
| Macro F1-score | 0.46 |
| 클래스별 F1 | 불만족: 0.09 / 보통: 0.63 / 만족: 0.66 |

→ 관련 수치: `final_model_classification_report.csv`, `final_model_confusion_matrix.csv`

---

## 클러스터별 개별 모델 vs 통합 모델 비교

GENDER, AGE_GRP 기준으로 데이터셋을 나누고  
클러스터별로 개별 모델을 학습하는 전략도 실행  

성능 비교 결과, 모든 클러스터에서 통합 모델이 Macro F1 기준 더 우수

Macro F1 기준 클러스터별 성능 비교 시각화: 'clusterwise_vs_final_model_comparison.png'

---

## 유사 클러스터 통합 예측 모델

성별, 연령대에 걸쳐 유사한 양상을 띈 클러스터끼리 묶어서 예측 모델 실행

성능 개선 실패
관련 결과 'similar_cluster_prediction_results.png'

---

## 디렉토리 구조

```
Classification/
├── README.md
├── data/
│   ├── preprocessed_with_cluster_ALL.csv        # 전권역 데이터
│   └── preprocessed_with_cluster_numbering.csv  # 클러스터 넘버링 포함
├── result/ 
│   ├── application_results.csv                  # 적용 예시 예측 결과
│   ├── clusterwise_vs_final_model_comparison.png # 클러스터별 학습 모델과 통합 모델 Macro_F1 비교
│   ├── final_model_classification_report.csv    
│   ├── final_model_confusion_matrix.csv         
│   └── similar_cluster_prediction_results.png   # 유사 클러스터 통합 예측 결과
├── src/  
│   ├── application_example.py                   # 적용 예시
│   ├── clusterwise_final_model.py               # 클러스터 별 최종 모델 성능
│   ├── clusterwise_predict_model.py             # 클러스터 별 개별 예측 모델
│   ├── final_model_pipeline.py                  
│   └── similar_cluster_predict_model.py         # 유사 클러스터 통합 예측 모델
```
