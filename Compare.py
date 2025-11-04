import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from tabpfn import TabPFNClassifier

df = pd.read_csv("student_perf_clean.csv")
y = df["label"].values
X = df.drop(columns=["label"]).values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

pfn = TabPFNClassifier().fit(Xtr, ytr)
pfn_probs = pfn.predict_proba(Xte)[:, 1]
pfn_preds = pfn.predict(Xte)

dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xtr, ytr)
dt_probs = dt.predict_proba(Xte)[:, 1]
dt_preds = dt.predict(Xte)

def safe_auc(y_true, p):
    try:
        return roc_auc_score(y_true, p)
    except ValueError:
        return float("nan")

print(f"TabPFN  AUC={safe_auc(yte, pfn_probs):.3f}  ACC={accuracy_score(yte, pfn_preds):.3f}")
print(f"DecisionTree AUC={safe_auc(yte, dt_probs):.3f}  ACC={accuracy_score(yte, dt_preds):.3f}")
