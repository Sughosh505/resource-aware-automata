import pandas as pd
df = pd.read_csv('master_dataset_final_3.csv')
print(df['optimal_model'].value_counts())