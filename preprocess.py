import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="raw CSV path (has 'label' column)")
    parser.add_argument("--out", dest="out_path", default="student_perf_clean.csv", help="clean CSV path")
    parser.add_argument("--winsorize", action="store_true", help="cap numeric outliers at 1%/99%")
    parser.add_argument("--label", dest="label_col", default="label", help="label column name")
    args = parser.parse_args()

    df = pd.read_csv(args.in_path)
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found.")

    print("=== RAW DATA CHECK ===")
    print("Shape:", df.shape)
    print("Top-5 missing values:\n", df.isna().sum().sort_values(ascending=False).head(5))
    y = df[args.label_col].values
    try:
        print("Class balance (mean of label):", float(np.mean(y)))
    except Exception:
        print("Class balance: (non-numeric label)")


    feature_cols = [c for c in df.columns if c != args.label_col]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]


    df_imp = df.copy()
    for c in num_cols:
        if df_imp[c].isna().any():
            df_imp[c] = df_imp[c].fillna(df_imp[c].median())
    for c in cat_cols:
        if df_imp[c].isna().any():
            mode_val = df_imp[c].mode(dropna=True)
            mode_val = mode_val.iloc[0] if len(mode_val) else ""
            df_imp[c] = df_imp[c].fillna(mode_val)

    if len(cat_cols) > 0:
        df_imp = pd.get_dummies(df_imp, columns=cat_cols, drop_first=True)


    if args.winsorize:
        clean_num_cols = [c for c in df_imp.columns if pd.api.types.is_numeric_dtype(df_imp[c]) and c != args.label_col]
        for c in clean_num_cols:
            lo, hi = df_imp[c].quantile(0.01), df_imp[c].quantile(0.99)
            df_imp[c] = df_imp[c].clip(lo, hi)


    df_imp.to_csv(args.out_path, index=False)


    print("\n=== CLEAN DATA CHECK ===")
    print("Shape:", df_imp.shape)
    print("Any missing left? ->", df_imp.isna().sum().sum())
    print("First 5 columns:", list(df_imp.columns[:5]))
    print(f"Saved clean file to: {args.out_path}")

if __name__ == "__main__":
    main()
