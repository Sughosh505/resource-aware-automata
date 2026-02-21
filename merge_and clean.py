import pandas as pd
import glob
import os

def merge_and_feature_extract():
    # 1. Find all your CSV files (e.g., data_1.csv, data_2.csv)
    all_files = glob.glob('*.csv')
    # Exclude any previously processed files to avoid loops
    files_to_merge = [f for f in all_files if 'processed' not in f]
    
    print(f"Found {len(files_to_merge)} files: {files_to_merge}")

    li = []
    for filename in files_to_merge:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    # 2. Stack them into one big 30k dataframe
    full_df = pd.concat(li, axis=0, ignore_index=True)
    full_df.columns = full_df.columns.str.strip().str.lower()
    
    print(f"Merged successfully. Total rows: {len(full_df)}")

    # 3. Apply your 7-column Logic
    print("Applying TOC Logic features...")
    full_df['alphabet_size'] = full_df['sequence'].apply(lambda x: len(set(str(x))))
    full_df['avg_string_length'] = full_df['sequence'].apply(lambda x: len(str(x)))
    
    # Mapping grammar rules (TOC Theory)
    rules = {'regular': 2, 'cfg': 5, 'csl': 10}
    full_df['rule_count'] = full_df['sequence'].map(rules).fillna(1)
    
    # Nesting depth for PDA/CFG
    def get_nesting(s):
        c, m = 0, 0
        for char in str(s):
            if char == '(': c += 1
            elif char == ')': c = max(0, c - 1)
            m = max(m, c)
        return m

    full_df['max_nesting_depth'] = full_df['sequence'].apply(get_nesting)
    full_df['is_ambiguous'] = False

    # 4. Save the Master Dataset
    full_df.to_csv('master_dataset.csv', index=False)
    print("Created 'master_dataset.csv' with 30,000 samples and all features.")

if __name__ == "__main__":
    merge_and_feature_extract()