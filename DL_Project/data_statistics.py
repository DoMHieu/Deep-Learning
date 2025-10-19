import pandas as pd

data = pd.read_csv("IMDB Dataset.csv")
print(data.describe())

print(data.dtypes)
