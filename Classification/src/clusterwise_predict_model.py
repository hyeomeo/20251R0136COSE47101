# -*- coding: utf-8 -*-
"""클러스터 별 개별 학습

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fcDByVpggBs3G7-hrdKZ3ElsLEWgZEik
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# 데이터 로딩
df = pd.read_csv("preprocessed_with_cluster_ALL.csv")
df = df.drop(columns=["REGION"], errors="ignore")

df = df.dropna(subset=[
    "DGSTFN", "VISIT_AREA_NM", "MAIN_TRAVEL_MONTH", "MVMN_NM",
    "REVISIT_INTENTION", "RCMDTN_INTENTION", "RELATION_TYPE",
    "VISIT_AREA_TYPE_CD"
])

# 클래스 정의: 1~3 불만족(0), 4 보통(1), 5 만족(2)
def map_label(score):
    if score == 5:
        return 2
    elif score in [3, 4]:
        return 1
    else:
        return 0

df["label_3cls"] = df["DGSTFN"].apply(map_label)

# 범주형/수치형 변수
cat_features = [
    "VISIT_AREA_NM", "MAIN_TRAVEL_MONTH", "TRAVEL_STATUS_ACCOMPANY",
    "RELATION_TYPE", "MVMN_NM", "GENDER", "AGE_GRP", "VISIT_AREA_TYPE_CD"
]
num_features = [
    "TRAVEL_STYL_1", "TRAVEL_STYL_3", "TRAVEL_STYL_5",
    "TRAVEL_STYL_6", "TRAVEL_STYL_7", "TRAVEL_STYL_8"
]

# 결과 저장용
group_results = []

# 클러스터별 분할 학습
for (gender, age), group_df in df.groupby(["GENDER", "AGE_GRP"]):
    if len(group_df) < 1000:
        continue  # 데이터가 너무 적은 그룹은 제외

    X = group_df[cat_features + num_features]
    y = group_df["label_3cls"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    # 전처리 + SMOTE + 분류기
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features)
    ])

    pipe = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(
            class_weight={0: 3, 1: 1, 2: 1},
            random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    group_results.append({
        "GENDER": gender,
        "AGE_GRP": age,
        "Samples": len(group_df),
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1_불만족": report["0"]["f1-score"],
        "F1_보통": report["1"]["f1-score"],
        "F1_만족": report["2"]["f1-score"],
        "Macro_F1": report["macro avg"]["f1-score"]
    })

# 결과 정리
results_df = pd.DataFrame(group_results)
print(results_df.sort_values("Macro_F1", ascending=False))
results_df.to_csv("clusterwise_rf_results.csv", index=False)
