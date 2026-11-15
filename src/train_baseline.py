import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

from features import load_dataset

df, X, y = load_dataset()

# Time-ordered split: the first 70% of each letter's recording trains the model,
# the last 30% tests it. A random split would put near-identical neighbouring
# frames in both sets and make accuracy look better than it really is.
train_idx, test_idx = [], []
for label in np.unique(y):
    idx = np.where(y == label)[0]
    cut = int(len(idx) * 0.7)
    train_idx += list(idx[:cut])
    test_idx += list(idx[cut:])

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVM (RBF)": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0),
}

best_name, best_acc, best_model = None, 0, None
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = (pred == y_test).mean()
    print(f"=== {name}: accuracy {acc:.3f}")
    print(classification_report(y_test, pred, digits=3))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test, pred, labels=np.unique(y)))
    print()
    if acc > best_acc:
        best_name, best_acc, best_model = name, acc, model

print(f"Best: {best_name} ({best_acc:.3f})")
joblib.dump(best_model, "models/baseline.joblib")