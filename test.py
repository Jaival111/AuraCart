import pandas as pd

df = pd.read_csv("amazon.csv")

print(df.category.unique())