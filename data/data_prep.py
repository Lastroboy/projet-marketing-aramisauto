import pandas as pd


df = pd.read_csv("./data/df_master.csv")

df["VILLE"] = df["VILLE"].str.upper()

df.to_csv("./data/df_master.csv")

