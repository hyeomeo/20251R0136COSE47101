import pandas as pd
import ast

# Pre-Processing
df = pd.read_csv("tn_visit_area_info_방문지정보_E.csv")
df.head(40)

# Show column index
print("Columns:", df.columns)
# Show row index
print("Index:", df.index)
# Show shape (rows, columns)
print("Shape:", df.shape)

df2 = df[:10000]
df2['area_len'] = df2['VISIT_AREA_NM'].apply(len)

# '집' 방문 제외
df2 = df2[df2['area_len'] > 1].reset_index(drop=True)
print(df2.shape)

df2.groupby('area_len').size().reset_index(name='count')
