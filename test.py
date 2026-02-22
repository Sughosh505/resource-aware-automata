import pandas as pd

df = pd.read_csv("master_dataset_final.csv")

# Only check numeric columns
numeric_df = df.select_dtypes(include="number")
print(numeric_df.corr()["optimal_model"].sort_values(ascending=False))