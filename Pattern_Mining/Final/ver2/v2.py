import pandas as pd

df = pd.read_csv("travel_data_final.csv")

cluster_map = {
     0: 0, 10: 0,                                                           # 그룹 0
     1: 1, 4: 1, 7: 1, 9: 1,                                                # 그룹 1
    13: 2, 15: 2, 20: 2,                                                    # 그룹 2
     2: 3, 3: 3, 8: 3, 11: 3, 14: 3, 16: 3, 19: 3, 23: 3, 25: 3, 28: 3      # 그룹 3
}

df["CLUSTER_V2"] = df["CLUSTER_NEW"].map(cluster_map)

df["CLUSTER_V2"] = df["CLUSTER_V2"].fillna(-1).astype(int)

df.to_csv("travel_data_final_v2.csv", index=False)
