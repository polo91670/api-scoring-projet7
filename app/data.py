import pandas as pd

def load_data(path="data.csv"):
    return pd.read_csv(path)