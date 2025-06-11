import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm

df = pd.read_csv("travel_data_final.csv")

font_path = 'C:/users/thetw/appdata/local/microsoft/Windows/Fonts/NanumGothic.ttf' # 폰트 경로 지정 필요

fm.fontManager.addfont(font_path)
fm.fontManager = fm.FontManager()

font_name = fm.FontProperties(fname=font_path).get_name()
mpl.rcParams['font.family'] = font_name
mpl.rcParams['font.sans-serif'] = [font_name]
mpl.rcParams['axes.unicode_minus'] = False

area_keep = df['VISIT_AREA_NM'].value_counts()
area_keep = area_keep[area_keep >= 50].index            # 50회 이상 지역만
df_filt   = df[df['VISIT_AREA_NM'].isin(area_keep)].copy()
dataset = df_filt.apply(
    lambda r: [f"CLUSTER_{r['CLUSTER_NEW']}", r['VISIT_AREA_NM']], axis=1
).tolist()

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df_tx  = pd.DataFrame(te_ary, columns=te.columns_)

support_grid = [0.001, 0.0005, 0.0003, 0.0001]   # 지지도 단계
rules_final  = pd.DataFrame()

for supp in support_grid:   # 클러스터 별 최소 패턴 가짓 수 보장
    freq  = fpgrowth(df_tx, min_support=supp, use_colnames=True)
    rules = association_rules(freq, metric='confidence', min_threshold=0.03)

    rules['CLUSTER_ID'] = rules['antecedents'].apply(
        lambda s: next((x for x in s if x.startswith('CLUSTER_')), None)
    )
    rules['CLUSTER_ID'] = rules['CLUSTER_ID'].fillna(
        rules['consequents'].apply(
            lambda s: next((x for x in s if x.startswith('CLUSTER_')), None)
        ),
    )
    cnt = rules['CLUSTER_ID'].value_counts()
    if (cnt >= 5).all():
        rules_final = rules
        break

rules_subset = rules_final[['antecedents', 'consequents', 'confidence']]

print(rules_final[['antecedents', 'consequents', 'support', 'confidence']])
rules_final[['antecedents', 'consequents', 'support', 'confidence']].to_csv("visitarea_cluster_rules.csv",index=False,encoding="utf-8-sig")   

# Initialize a directed graph
G = nx.DiGraph()

# Prepare color mapping based on confidence values
confidences = rules_subset['confidence'].tolist()
min_conf, max_conf = min(confidences), max(confidences)
norm = mcolors.Normalize(vmin=min_conf, vmax=max_conf)
cmap = plt.get_cmap('RdYlGn_r')  # Red (low) → Green (high)

# Add directed edges to the graph
for _, row in rules_subset.iterrows():
    antecedent = list(row['antecedents'])[0]
    consequent = list(row['consequents'])[0]
    confidence = row['confidence']
    color = cmap(norm(confidence))  # Map confidence to color
    G.add_edge(antecedent, consequent, confidence=confidence, color=color)

# Set up plot
fig, ax = plt.subplots(figsize=(10, 8))

# Layout for node positions; smaller k = tighter layout
pos = nx.spring_layout(G, seed=42, k=0.5)

# Extract edge colors from confidence values
edge_colors = [G[u][v]['color'] for u, v in G.edges()]

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)

# Draw directed edges with curve to reduce overlap
nx.draw_networkx_edges(
    G, pos, ax=ax, edge_color=edge_colors,
    arrows=True, arrowstyle='-|>', arrowsize=30, width=2,
    connectionstyle='arc3,rad=0.2'  # Slight curve for overlapping arrows
)

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax)

# Draw edge labels showing confidence values
edge_labels = {(u, v): f"{G[u][v]['confidence']:.2f}" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, ax=ax)

# Add color bar for confidence scale
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
cbar.set_label('Confidence')

plt.title('VISIT_AREA_NM <-> CLUSTER  (지역 ≥ 50회, FP-Growth)', fontsize=14)
plt.axis('off');
plt.tight_layout();
plt.show()
