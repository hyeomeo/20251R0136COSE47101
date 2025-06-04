
from common import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data with and without cluster
df_with = load_dataset(region="E", with_cluster=True)
df_without = load_dataset(region="E", with_cluster=False)

for df, label in zip([df_without, df_with], ["No Cluster", "With Cluster"]):
    df["label"] = (df["DGSTFN"] >= 4.0).astype(int)
    X = df[["VISIT_AREA_NM", "MAIN_TRAVEL_MONTH", "REVISIT_INTENTION", "RCMDTN_INTENTION"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["VISIT_AREA_NM"])
    ], remainder="passthrough")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n📊 Logistic Regression Classification Report ({label})")
    print(classification_report(y_test, y_pred))
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))
    print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
