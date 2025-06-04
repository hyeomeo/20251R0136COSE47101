import pandas as pd

df = pd.read_csv("Clustering/0602/data/clustered_data/clustered_k=3_per_gender&age.csv")

gender_list = [0, 1]
age_grp_list = [0, 0.25, 0.5, 0.75, 1.0]
cluster_list = [0, 1, 2]

mapping = {}
new_cluster_num = 0

for age in age_grp_list:
    for gender in gender_list:
        for cluster in cluster_list:
            key = (gender, age, cluster)
            mapping[key] = new_cluster_num
            new_cluster_num += 1   # 0~29로 매핑

df['CLUSTER_NEW'] = df.apply(lambda row: mapping.get((row['GENDER'], row['AGE_GRP'], row['CLUSTER'])), axis=1)

df.to_csv('Clustering/0602/data/clustered_data/updated_clustered_data_total_k30.csv', index=False)