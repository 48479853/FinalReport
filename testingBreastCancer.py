from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# TabPFN: single fit + predict (no tuning)
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Probabilities and labels
probs = clf.predict_proba(X_test)[:, 1]
preds = clf.predict(X_test)

print("ROC AUC:", roc_auc_score(y_test, probs))
print("Accuracy:", accuracy_score(y_test, preds))
