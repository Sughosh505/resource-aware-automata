import pandas as pd
import os

def extract_features(s, lang_name):
    """Calculates the 7 core features for a single string."""
    s = str(s)
    # 1. language_name (already provided)
    # 2. alphabet_size
    alphabet_size = len(set(s))
    # 3. rule_count (Theoretical mapping)
    rules = {'parity': 2, 'dyck': 5, 'anbncn': 10}
    rule_count = rules.get(lang_name, 1)
    # 4. max_nesting_depth
    depth, max_d = 0, 0
    for char in s:
        if char == '(': depth += 1
        elif char == ')': depth = max(0, depth - 1)
        max_d = max(max_d, depth)
    # 5. avg_string_length
    length = len(s)
    # 6. is_ambiguous (Static for these languages)
    is_ambiguous = 0 
    # 7. sequence_complexity (entropy-like measure)
    complexity = alphabet_size / length if length > 0 else 0

    return alphabet_size, rule_count, max_d, length, is_ambiguous, complexity

def run_full_pipeline():
    files = {
        'data_parity.csv': 'parity',
        'data_dyck.csv': 'dyck',
        'data_abc.csv': 'anbncn'
    }
    
    all_data = []

    for filename, lang in files.items():
        if not os.path.exists(filename):
            print(f"Skipping {filename}: Not found.")
            continue
            
        print(f"Processing {filename}...")
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip().str.lower()
        
        # Ensure the language name is set correctly
        df['language_name'] = lang
        
        # Apply the feature extraction to all 10k rows in this file
        features = df.apply(lambda row: extract_features(row['sequence'], lang), axis=1)
        
        # Split the tuple result into the 7 columns
        (df['alphabet_size'], df['rule_count'], df['max_nesting_depth'], 
         df['avg_string_length'], df['is_ambiguous'], df['complexity']) = zip(*features)
        
        all_data.append(df)

    if all_data:
        # Merge all dataframes
        master_df = pd.concat(all_data, ignore_index=True)
        # Shuffle the 30k-40k rows
        master_df = master_df.sample(frac=1).reset_index(drop=True)
        # Save
        master_df.to_csv('master_dataset.csv', index=False)
        print("\nSuccess! master_dataset.csv created with all columns.")
        print(master_df[['language_name', 'label']].value_counts())
    else:
        print("No data was processed.")

if __name__ == "__main__":
    run_full_pipeline()