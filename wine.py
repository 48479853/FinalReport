# run_wine.py — TabPFN vs DecisionTree on Wine (3 classes)
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier
import numpy as np

def ovr_auc(y_true, proba, classes):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    classes = np.asarray(classes)
    if proba.ndim == 1 or proba.shape[1] == 1:
        return float("nan")
    y_bin = label_binarize(y_true, classes=classes)
    try:
        return roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
    except ValueError:
        return float("nan")

def main():
    data = load_wine()
    X, y = data.data, data.target
    classes = np.unique(y)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # TabPFN
    pfn = TabPFNClassifier().fit(Xtr, ytr)
    pfn_proba = pfn.predict_proba(Xte)
    pfn_pred  = pfn.predict(Xte)
    pfn_acc   = accuracy_score(yte, pfn_pred)
    pfn_auc   = ovr_auc(yte, pfn_proba, classes)

    # Decision Tree baseline
    dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xtr, ytr)
    dt_proba = dt.predict_proba(Xte)
    dt_pred  = dt.predict(Xte)
    dt_acc   = accuracy_score(yte, dt_pred)
    dt_auc   = ovr_auc(yte, dt_proba, classes)

    print("Wine (multiclass) — 70/30 split, seed=42")
    print(f"TabPFN       ACC={pfn_acc:.3f}  AUC(OVR)={pfn_auc:.3f}")
    print(f"DecisionTree ACC={dt_acc:.3f}  AUC(OVR)={dt_auc:.3f}")

if __name__ == "__main__":
    main()
