
from common import load_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

for cluster in [True, False]:
    df = load_dataset(region="E", with_cluster=cluster)
    X = df[["VISIT_AREA_NM", "MAIN_TRAVEL_MONTH", "REVISIT_INTENTION", "RCMDTN_INTENTION",
            "GENDER", "AGE_GRP", "TRAVEL_STYL_1", "TRAVEL_STYL_3", "TRAVEL_STYL_5",
            "TRAVEL_STYL_6", "TRAVEL_STYL_7", "TRAVEL_STYL_8"] + (["CLUSTER"] if cluster else [])]
    y = df["DGSTFN"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    cat_cols = ["VISIT_AREA_NM", "GENDER", "AGE_GRP"] + (["CLUSTER"] if cluster else [])
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder="passthrough")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"\n📊 RandomForestRegressor Results ({'With' if cluster else 'No'} Cluster)")
    print("✅ MAE:", round(mean_absolute_error(y_test, preds), 4))
    print("✅ MSE:", round(mean_squared_error(y_test, preds), 4))
    print("✅ R²:", round(r2_score(y_test, preds), 4))
