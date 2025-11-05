import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)


try:
    from tabpfn import TabPFNClassifier
    HAVE_TABPFN = True
except Exception:
    TabPFNClassifier = None
    HAVE_TABPFN = False


def coerce_binary(y):
    """
    Make sure labels are 0/1. Supports 'pass'/'fail', 'yes'/'no', etc.
    """
    y = pd.Series(y)
    if set(y.unique()) <= {0, 1, 0.0, 1.0}:
        return y.astype(int).values

    # Common strings
    lower = y.astype(str).str.lower()
    maps = [
        {"fail": 0, "pass": 1},
        {"no": 0, "yes": 1},
        {"negative": 0, "positive": 1},
        {"false": 0, "true": 1},
    ]
    for m in maps:
        if set(lower.unique()) <= set(m.keys()):
            return lower.map(m).astype(int).values

    vals = sorted(lower.unique())
    if len(vals) == 2:
        print(f"[WARN] Unknown labels {vals}. Mapping {vals[0]}->0, {vals[1]}->1")
        m = {vals[0]: 0, vals[1]: 1}
        return lower.map(m).astype(int).values

    raise ValueError("Label column is not binary or cannot be coerced to {0,1}.")


def safe_auc(y_true, probs):
    try:
        return roc_auc_score(y_true, probs)
    except Exception:
        return float("nan")


def print_metrics(name, y_true, probs, preds):
    auc = safe_auc(y_true, probs)
    acc = accuracy_score(y_true, preds)
    f1  = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    cm   = confusion_matrix(y_true, preds)

    print(f"\n=== {name} : Test Set Metrics ===")
    print(f"AUC:       {auc:0.3f}")
    print(f"Accuracy:  {acc:0.3f}")
    print(f"F1:        {f1:0.3f}")
    print(f"Precision: {prec:0.3f}")
    print(f"Recall:    {rec:0.3f}")
    print("Confusion Matrix [ [TN FP] ; [FN TP] ]:")
    print(cm)


def print_cv(name, model, X, y, seed=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    accs = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\n--- {name} : Stratified 5-fold CV ---")
    print(f"AUC: mean={aucs.mean():0.3f}  std={aucs.std():0.3f}")
    print(f"ACC: mean={accs.mean():0.3f}  std={accs.std():0.3f}")


def main():
    ap = argparse.ArgumentParser(description="TabPFN vs DecisionTree on student performance (terminal output only)")
    ap.add_argument("--data", default="student_perf_clean.csv", help="CSV path")
    ap.add_argument("--label", default="label", help="Label column name")
    ap.add_argument("--test_size", type=float, default=0.30, help="Test fraction (default 0.30)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.label not in df.columns:
        raise ValueError(f"Label column '{args.label}' not in columns: {list(df.columns)}")

    y = coerce_binary(df[args.label].values)
    X = df.drop(columns=[args.label]).values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    tree = DecisionTreeClassifier(max_depth=3, random_state=args.seed).fit(Xtr, ytr)
    tree_probs = tree.predict_proba(Xte)[:, 1]
    tree_preds = tree.predict(Xte)
    print_metrics("DecisionTree (baseline)", yte, tree_probs, tree_preds)
    print_cv("DecisionTree (baseline)", tree, X, y, seed=args.seed)

    if HAVE_TABPFN:
        pfn = TabPFNClassifier().fit(Xtr, ytr)
        pfn_probs = pfn.predict_proba(Xte)[:, 1]
        pfn_preds = pfn.predict(Xte)
        print_metrics("TabPFN", yte, pfn_probs, pfn_preds)
        print_cv("TabPFN", TabPFNClassifier(), X, y, seed=args.seed)
    else:
        print("\n[INFO] TabPFN not installed. Install with:")
        print("       pip install 'tabpfn @ git+https://github.com/PriorLabs/TabPFN.git'")


if __name__ == "__main__":
    main()
