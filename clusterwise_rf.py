
from common import load_dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

df = load_dataset(region="E", with_cluster=True)
results = []

for cluster_id in sorted(df["CLUSTER"].unique()):
    sub_df = df[df["CLUSTER"] == cluster_id]
    X = sub_df[["VISIT_AREA_NM", "MAIN_TRAVEL_MONTH", "REVISIT_INTENTION", "RCMDTN_INTENTION",
                "GENDER", "AGE_GRP", "TRAVEL_STYL_1", "TRAVEL_STYL_3", "TRAVEL_STYL_5",
                "TRAVEL_STYL_6", "TRAVEL_STYL_7", "TRAVEL_STYL_8"]]
    y = sub_df["DGSTFN"]

    if len(sub_df) < 30:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(pd.get_dummies(X_train), y_train)
    preds = model.predict(pd.get_dummies(X_test))

    results.append({
        "CLUSTER": int(cluster_id),
        "MAE": round(mean_absolute_error(y_test, preds), 4),
        "MSE": round(mean_squared_error(y_test, preds), 4),
        "R2": round(r2_score(y_test, preds), 4)
    })

df_result = pd.DataFrame(results)
print("\n📊 Cluster-wise RandomForestRegressor Results")
print(df_result)
