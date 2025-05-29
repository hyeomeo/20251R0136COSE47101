# Baseline logistic regression for satisfaction prediction

import pandas as pd
df = pd.read_csv("/content/cleaned_dataset.csv")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = df.dropna(subset=["DGSTFN", "VISIT_AREA_NM", "VISIT_START_YMD", "REVISIT_INTENTION", "RCMDTN_INTENTION"])

df["label"] = (df["DGSTFN"] >= 4.0).astype(int)
df["month"] = df["VISIT_START_YMD"].astype(str).str[5:7].astype(int)

# 피처 및 타겟 정의
X = df[["VISIT_AREA_NM", "month", "REVISIT_INTENTION", "RCMDTN_INTENTION"]]
y = df["label"]

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 전처리 및 모델 파이프라인 구성
categorical_features = ["VISIT_AREA_NM"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="passthrough"
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

# 모델 학습 및 예측
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 평가 결과 출력
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print("=== Classification Report ===")
print(pd.DataFrame(report).transpose())
print("\n=== Confusion Matrix ===")
print(conf_matrix)
print(f"\nAccuracy: {accuracy:.4f}")
