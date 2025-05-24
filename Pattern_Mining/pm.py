import ast
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import networkx as nx
import matplotlib as mpl

df = pd.read_csv("tn_travel_여행_E.csv")

font_path = 'C:/Windows/Fonts/NanumGothic.ttf'
fm.fontManager.addfont(font_path)
fm.fontManager = fm.FontManager()

font_name = fm.FontProperties(fname=font_path).get_name()
mpl.rcParams['font.family'] = font_name
mpl.rcParams['font.sans-serif'] = [font_name]
mpl.rcParams['axes.unicode_minus'] = False

# Show column index
print("Columns:", df.columns)

# Show row index
print("Index:", df.index)

# Show shape (rows, columns)
print("Shape:", df.shape)

df2 = df[:2000]

def build_option(text):
    parts = text.split('/')
    base  = parts[:3]
#    if '미션' not in parts[3]:
#        base.append(parts[3])
    return ', '.join(base)

df2['option'] = df2['TRAVEL_PERSONA'].apply(build_option)

words = []
for i, word in enumerate(df2['option']):
    if i >= 2000:
        break
    words.append(word)

dataset = [ [item.strip() for item in w.split(',')] for w in words ]
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df, min_support=0.008, use_colnames=True)
# print(frequent_itemsets)

frequent_itemsets_multi = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) >= 2)]
print(frequent_itemsets_multi)

# Extract necessary data
supports = frequent_itemsets_multi['support'].tolist()
itemsets = frequent_itemsets_multi['itemsets'].tolist()

# Calculate minimum and maximum support values
min_support, max_support = min(supports), max(supports)

# Function to scale support values into edge widths
def scale_support(support, min_val, max_val, min_width=1, max_width=6):
    return min_width + (support - min_val) / (max_val - min_val) * (max_width - min_width)

# Create a graph
G = nx.Graph()

# Add edges from the data
for support, itemset in zip(supports, itemsets):
    items = list(itemset)
    if len(items) == 2:  # Only add pairs
        G.add_edge(items[0], items[1], weight=support, width=scale_support(support, min_support, max_support))


# Visualize the graph
plt.figure(figsize=(12, 9))
pos = nx.spring_layout(G, seed=42)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=600, node_color='red')

# Draw edges with varying widths
edge_widths = [G[u][v]['width'] for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color='gray')

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', font_color='black')

# Draw edge labels (support values included)
edge_labels = {(u, v): f"{G[u][v]['weight']:.4f} (support)" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title('Word graph with support values', fontsize=20)
plt.axis('off')
plt.show()

# Generate rules with min confidence=70%
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])

import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Select relevant columns from the association rules
rules_subset = rules[['antecedents', 'consequents', 'confidence']]

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

# Final touches
plt.axis('off')
plt.tight_layout()
plt.show()
