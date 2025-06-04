
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(region="E", with_cluster=True):
    base = f"data/{region}/"
    filename = f"preprocessed_with_cluster_{region}.csv" if with_cluster else f"preprocessed_{region}.csv"
    df = pd.read_csv(base + filename)
    return df.dropna(subset=["DGSTFN"])
