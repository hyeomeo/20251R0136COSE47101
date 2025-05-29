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

X = df[["VISIT_AREA_NM", "month", "REVISIT_INTENTION", "RCMDTN_INTENTION"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

categorical_features = ["VISIT_AREA_NM"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="passthrough"
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print("=== Classification Report ===")
print(pd.DataFrame(report).transpose())
print("\n=== Confusion Matrix ===")
print(conf_matrix)
print(f"\nAccuracy: {accuracy:.4f}")
