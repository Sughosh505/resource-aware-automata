"""
XGBoost Classifier - Optimal Automaton Model Predictor
Predicts: 0 = DFA, 1 = PDA, 2 = Turing Machine (based on energy efficiency)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = "master_dataset_noisy_1.csv"
TARGET_COL  = "optimal_model"
DROP_COLS = ["sequence", "label", "language_name"]
CLASS_NAMES = {0: "DFA", 1: "PDA", 2: "Turing Machine"}
RANDOM_SEED = 42
TEST_SIZE   = 0.2

# Features the USER will provide at prediction time (first 7 feature columns)
# Energy/state columns are excluded — model fills them with training-set medians
USER_INPUT_FEATURES = [
    "alphabet_size",
    "rule_count",
    "max_nesting_depth",
    "avg_string_length",
    "is_ambiguous",
    "complexity",
    "dfa_state",
]

FEATURE_HINTS = {
    "alphabet_size"     : "e.g. 2 (binary), 26 (English letters)",
    "rule_count"        : "number of grammar / transition rules",
    "max_nesting_depth" : "max depth of nested structures (0 = none)",
    "avg_string_length" : "average input string length",
    "is_ambiguous"      : "0 = No, 1 = Yes",
    "complexity"        : "numerical complexity score",
    "dfa_state"         : "number of DFA states",
}


# ─────────────────────────────────────────────
#  1. LOAD & PREPROCESS
# ─────────────────────────────────────────────
def load_and_preprocess(path):
    print("\n📂  Loading dataset...")
    df = pd.read_csv(path)
    print(f"   Shape     : {df.shape}")
    print(f"   Columns   : {list(df.columns)}\n")

    # Encode any remaining categorical text columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        if col not in DROP_COLS and col != TARGET_COL:
            df[col] = le.fit_transform(df[col].astype(str))

    # Drop identifier columns
    drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=drop, inplace=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"   Features  : {list(X.columns)}")
    print(f"   Class dist:\n{y.value_counts().rename(CLASS_NAMES)}\n")
    return X, y


# ─────────────────────────────────────────────
#  2. TRAIN
# ─────────────────────────────────────────────
def train_model(X_train, y_train):
    print("🚀  Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.1,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        use_label_encoder = False,
        eval_metric       = "mlogloss",
        random_state      = RANDOM_SEED,
        n_jobs            = -1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)
    print("   ✅  Training complete.\n")
    return model


# ─────────────────────────────────────────────
#  3. EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("=" * 55)
    print(f"  🎯  TEST ACCURACY : {acc * 100:.2f}%")
    print("=" * 55)
    print("\n📊  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    ))
    return y_pred, acc


# ─────────────────────────────────────────────
#  4. PLOT RESULTS
# ─────────────────────────────────────────────
def plot_results(model, X_test, y_test, y_pred, acc, feature_names):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("XGBoost — Optimal Automaton Model Classifier", fontsize=16, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

    # ── (A) Confusion Matrix ──────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    cm  = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax1, colorbar=False, cmap="Blues")
    ax1.set_title("Confusion Matrix", fontweight="bold")
    ax1.tick_params(axis="x", rotation=15)

    # ── (B) Feature Importance ────────────────
    ax2 = fig.add_subplot(gs[0, 1:])
    importances = model.feature_importances_
    idx         = np.argsort(importances)[::-1]
    top_n       = min(12, len(feature_names))
    colors      = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax2.barh(
        [feature_names[i] for i in idx[:top_n]][::-1],
        importances[idx[:top_n]][::-1],
        color=colors
    )
    ax2.set_xlabel("Importance Score")
    ax2.set_title("Top Feature Importances", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    # ── (C) Class Distribution (True vs Pred) ─
    ax3 = fig.add_subplot(gs[1, 0])
    true_counts = pd.Series(y_test).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    x     = np.arange(len(labels))
    width = 0.35
    ax3.bar(x - width/2, [true_counts.get(i, 0) for i in range(3)], width, label="Actual",    color="#4C72B0", alpha=0.85)
    ax3.bar(x + width/2, [pred_counts.get(i, 0) for i in range(3)], width, label="Predicted", color="#DD8452", alpha=0.85)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=10)
    ax3.set_ylabel("Count"); ax3.set_title("Actual vs Predicted Distribution", fontweight="bold")
    ax3.legend(); ax3.grid(axis="y", alpha=0.3)

    # ── (D) Per-class Precision / Recall ──────
    ax4 = fig.add_subplot(gs[1, 1])
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2])
    ax4.plot(labels, prec, "o-",  label="Precision", color="#2ca02c")
    ax4.plot(labels, rec,  "s--", label="Recall",    color="#d62728")
    ax4.plot(labels, f1,   "^:",  label="F1-Score",  color="#9467bd")
    ax4.set_ylim(0, 1.05); ax4.set_ylabel("Score")
    ax4.set_title("Per-class Metrics", fontweight="bold")
    ax4.legend(); ax4.grid(alpha=0.3)

    # ── (E) Accuracy Badge ────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    ax5.text(0.5, 0.6, f"{acc*100:.2f}%", ha="center", va="center",
             fontsize=48, fontweight="bold", color="#1a7f3c",
             transform=ax5.transAxes)
    ax5.text(0.5, 0.35, "Test Accuracy", ha="center", va="center",
             fontsize=16, color="#444", transform=ax5.transAxes)
    ax5.text(0.5, 0.18, f"({len(y_test):,} test samples)", ha="center", va="center",
             fontsize=11, color="#888", transform=ax5.transAxes)
    rect = plt.Rectangle((0.05, 0.1), 0.9, 0.75, fill=False,
                          edgecolor="#1a7f3c", linewidth=2.5, transform=ax5.transAxes)
    ax5.add_patch(rect)

    plt.savefig("xgboost_results.png", dpi=150, bbox_inches="tight")
    print("   📈  Plot saved → xgboost_results.png\n")
    plt.show()


# ─────────────────────────────────────────────
#  5. INTERACTIVE PREDICTION
#     User only provides the 7 structural
#     features. Energy/stack columns the user
#     can't know are filled with training-set
#     medians as a safe neutral substitute.
# ─────────────────────────────────────────────
def predict_custom(model, feature_names, X_train):
    print("\n" + "=" * 55)
    print("  🔍  CUSTOM INPUT PREDICTOR")
    print("      (energy columns auto-filled by model)")
    print("=" * 55)
    print("Enter values for each feature (or press Enter for 0).\n")

    # Pre-compute medians from training set for hidden features
    hidden_features = [f for f in feature_names if f not in USER_INPUT_FEATURES]
    medians = X_train[hidden_features].median().to_dict() if hidden_features else {}

    if hidden_features:
        print(f"   ℹ️  Auto-filled from training data : {hidden_features}\n")

    # Collect user inputs
    user_values = {}
    for feat in USER_INPUT_FEATURES:
        if feat not in feature_names:
            continue
        hint     = FEATURE_HINTS.get(feat, "")
        hint_str = f"  [{hint}]" if hint else ""
        raw      = input(f"  {feat}{hint_str}: ").strip()
        try:
            user_values[feat] = float(raw) if raw else 0.0
        except ValueError:
            print(f"   ⚠️  Invalid input for '{feat}', defaulting to 0.")
            user_values[feat] = 0.0

    # Build full feature row in the correct column order
    full_row = {}
    for feat in feature_names:
        if feat in user_values:
            full_row[feat] = user_values[feat]
        else:
            full_row[feat] = medians.get(feat, 0.0)

    input_df   = pd.DataFrame([full_row])
    pred_class = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]

    print("\n" + "─" * 45)
    print(f"  🏆  PREDICTED OPTIMAL MODEL : {CLASS_NAMES[pred_class]}  (class {pred_class})")
    print("─" * 45)
    print("  Confidence scores:")
    for i, p in enumerate(proba):
        bar = "█" * int(p * 30)
        print(f"    {CLASS_NAMES[i]:<16}: {p*100:5.1f}%  {bar}")
    print()

    # Ambiguity warning
    if user_values.get("is_ambiguous", 0) == 1:
        print("  ⚠️  Note: Language is AMBIGUOUS — DFA cannot be optimal.")
        if pred_class == 0:
            runner_up = int(np.argsort(proba)[-2])
            print(f"     Consider the runner-up: {CLASS_NAMES[runner_up]} ({proba[runner_up]*100:.1f}%)")
    print()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load & preprocess
    X, y = load_and_preprocess(DATA_PATH)
    feature_names = list(X.columns)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    # 3. Train
    model = train_model(X_train, y_train)

    # 4. Evaluate
    y_pred, acc = evaluate(model, X_test, y_test)

    # 5. Plot
    print("📊  Generating plots...")
    plot_results(model, X_test, y_test, y_pred, acc, feature_names)

    # 6. Interactive prediction loop
    while True:
        again = input("🔁  Test a custom input? (y/n): ").strip().lower()
        if again == "y":
            predict_custom(model, feature_names, X_train)
        else:
            print("\n✅  Done! Goodbye.\n")
            break