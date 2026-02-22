

import pandas as pd
import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
df = pd.read_csv("master_dataset_final.csv")
print(f"Original shape: {df.shape}")
print(f"Original class distribution:\n{df['optimal_model'].value_counts()}\n")

noisy = df.copy()

# ─────────────────────────────────────────────
#  1. GAUSSIAN NOISE on numeric columns
#     Reduced from 15% → 5% std
# ─────────────────────────────────────────────
numeric_cols = [
    "avg_string_length",
    "max_nesting_depth",
    "complexity",
    "rule_count",
    "alphabet_size",
    "dfa_energy",
    "pda_energy",
    "dfa_state",
    "pda_stack",
]

for col in numeric_cols:
    if col not in noisy.columns:
        continue
    std   = noisy[col].std()
    noise = np.random.normal(0, std * 0.05, size=len(noisy))  # 5% (was 15%)
    noisy[col] = noisy[col] + noise
    if col not in ["complexity"]:
        noisy[col] = noisy[col].clip(lower=0)

print("✅  Gaussian noise added to numeric columns (5% std)")

# ─────────────────────────────────────────────
#  2. FLIP is_ambiguous RANDOMLY (~3% of rows)
#     Reduced from 8% → 3%
# ─────────────────────────────────────────────
flip_mask = np.random.random(len(noisy)) < 0.03
noisy.loc[flip_mask, "is_ambiguous"] = 1 - noisy.loc[flip_mask, "is_ambiguous"]
print(f"✅  Flipped is_ambiguous on {flip_mask.sum()} rows (~3%)")

# ─────────────────────────────────────────────
#  3. LABEL NOISE — flip optimal_model (~2%)
#     Reduced from 5% → 2%
#     This is the key lever for accuracy control
# ─────────────────────────────────────────────
label_flip_mask = np.random.random(len(noisy)) < 0.02
original_labels = noisy.loc[label_flip_mask, "optimal_model"].values

def flip_label(label):
    choices = [x for x in [0, 1, 2] if x != label]
    return np.random.choice(choices)

noisy.loc[label_flip_mask, "optimal_model"] = [flip_label(l) for l in original_labels]
print(f"✅  Flipped optimal_model label on {label_flip_mask.sum()} rows (~2%)")

# ─────────────────────────────────────────────
#  4. MISSING VALUES → filled with median (~1%)
#     Reduced from 3% → 1%
# ─────────────────────────────────────────────
for col in numeric_cols:
    if col not in noisy.columns:
        continue
    missing_mask = np.random.random(len(noisy)) < 0.01
    noisy.loc[missing_mask, col] = np.nan

noisy[numeric_cols] = noisy[numeric_cols].apply(lambda c: c.fillna(c.median()))
print(f"✅  Injected and filled missing values (~1% per numeric column)")

# ─────────────────────────────────────────────
#  5. OUTLIERS — inject extreme values (~0.5%)
#     Reduced from 1% → 0.5%
# ─────────────────────────────────────────────
for col in ["avg_string_length", "complexity", "rule_count"]:
    if col not in noisy.columns:
        continue
    outlier_mask = np.random.random(len(noisy)) < 0.005
    multiplier   = np.random.uniform(2, 4, size=outlier_mask.sum())  # milder (was 3-6x)
    noisy.loc[outlier_mask, col] = noisy.loc[outlier_mask, col] * multiplier
print(f"✅  Injected outliers (~0.5%, milder multiplier)\n")

# ─────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────
output_path = "master_dataset_noisy_1.csv"
noisy.to_csv(output_path, index=False)
"""
Add real-world noise to synthetic dataset
Creates a noisier version of master_dataset_final.csv
"""

import pandas as pd
import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
df = pd.read_csv("master_dataset_final.csv")
print(f"Original shape: {df.shape}")
print(f"Original class distribution:\n{df['optimal_model'].value_counts()}\n")

noisy = df.copy()

# ─────────────────────────────────────────────
#  1. GAUSSIAN NOISE on numeric columns
#     Adds small random fluctuations — simulates
#     measurement error / real-world variance
# ─────────────────────────────────────────────
numeric_cols = [
    "avg_string_length",
    "max_nesting_depth",
    "complexity",
    "rule_count",
    "alphabet_size",
    "dfa_energy",
    "pda_energy",
    "dfa_state",
    "pda_stack",
]

for col in numeric_cols:
    if col not in noisy.columns:
        continue
    std   = noisy[col].std()
    noise = np.random.normal(0, std * 0.15, size=len(noisy))  # 15% std noise
    noisy[col] = noisy[col] + noise
    # clip to non-negative where it makes sense
    if col not in ["complexity"]:
        noisy[col] = noisy[col].clip(lower=0)

print("✅  Gaussian noise added to numeric columns")

# ─────────────────────────────────────────────
#  2. FLIP is_ambiguous RANDOMLY (~8% of rows)
#     Real grammars are sometimes misclassified
#     for ambiguity — this simulates that
# ─────────────────────────────────────────────
flip_mask = np.random.random(len(noisy)) < 0.08
noisy.loc[flip_mask, "is_ambiguous"] = 1 - noisy.loc[flip_mask, "is_ambiguous"]
print(f"✅  Flipped is_ambiguous on {flip_mask.sum()} rows (~8%)")

# ─────────────────────────────────────────────
#  3. LABEL NOISE — flip optimal_model (~5%)
#     Simulates expert disagreement, edge cases,
#     or borderline languages where two models
#     perform nearly equally
# ─────────────────────────────────────────────
label_flip_mask = np.random.random(len(noisy)) < 0.05
original_labels = noisy.loc[label_flip_mask, "optimal_model"].values

def flip_label(label):
    choices = [x for x in [0, 1, 2] if x != label]
    return np.random.choice(choices)

noisy.loc[label_flip_mask, "optimal_model"] = [flip_label(l) for l in original_labels]
print(f"✅  Flipped optimal_model label on {label_flip_mask.sum()} rows (~5%)")

# ─────────────────────────────────────────────
#  4. MISSING VALUES → filled with median (~3%)
#     Real datasets always have some missing data
# ─────────────────────────────────────────────
for col in numeric_cols:
    if col not in noisy.columns:
        continue
    missing_mask = np.random.random(len(noisy)) < 0.03
    noisy.loc[missing_mask, col] = np.nan

noisy[numeric_cols] = noisy[numeric_cols].apply(lambda c: c.fillna(c.median()))
print(f"✅  Injected and filled missing values (~3% per numeric column)")

# ─────────────────────────────────────────────
#  5. OUTLIERS — inject extreme values (~1%)
#     Simulates rare but real edge-case languages
# ─────────────────────────────────────────────
for col in ["avg_string_length", "complexity", "rule_count"]:
    if col not in noisy.columns:
        continue
    outlier_mask = np.random.random(len(noisy)) < 0.01
    multiplier   = np.random.uniform(3, 6, size=outlier_mask.sum())
    noisy.loc[outlier_mask, col] = noisy.loc[outlier_mask, col] * multiplier
print(f"✅  Injected outliers into avg_string_length, complexity, rule_count (~1%)")

# ─────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────
output_path = "master_dataset_noisy.csv"
noisy.to_csv(output_path, index=False)

print(f"\n📂  Saved → {output_path}")
print(f"   Shape: {noisy.shape}")
print(f"   New class distribution:\n{noisy['optimal_model'].value_counts()}\n")
print("Done! Use master_dataset_noisy.csv in your XGBoost script.")
print(f"📂  Saved → {output_path}")
print(f"   Shape: {noisy.shape}")
print(f"   New class distribution:\n{noisy['optimal_model'].value_counts()}\n")
print("Done! Use master_dataset_noisy.csv in your XGBoost script.")
print("\n💡  Tip: If accuracy is still above 98%, increase label noise to 3%.")
print("    If accuracy drops below 96%, reduce label noise back to 1.5%.")