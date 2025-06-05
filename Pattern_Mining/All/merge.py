import glob
import pandas as pd

file_list = glob.glob("preprocessed_*.csv")


df = pd.concat(
        (pd.read_csv(f) for f in file_list),
        ignore_index=True
     )

df.to_csv(r"./travel_data_merged.csv", index=False)
