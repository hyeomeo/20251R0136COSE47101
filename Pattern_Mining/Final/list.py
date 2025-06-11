import pandas as pd
import re

df = pd.read_csv('visitarea_cluster_rules.csv')

def extract_values(fset_str):
    return re.findall(r"'(.*?)'", fset_str)

df['antecedents'] = df['antecedents'].apply(extract_values)
df['consequents'] = df['consequents'].apply(extract_values)

cluster_dict = {f'CLUSTER_{i}': set() for i in range(30)}

for _, row in df.iterrows():
    items = row['antecedents'] + row['consequents']
    clusters = [x for x in items if x.startswith('CLUSTER_')]
    areas = [x for x in items if not x.startswith('CLUSTER_')]
    for cluster in clusters:
        cluster_dict[cluster].update(areas)

output_lines = []

for cluster in sorted(cluster_dict.keys(), key=lambda x: int(x.split('_')[1])):
    output_lines.append(f"{cluster}:\n")
    for area in sorted(cluster_dict[cluster]):
        output_lines.append(f"    {area}\n")
    output_lines.append("\n")

output_path = "cluster_transaction_output.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.writelines(output_lines)