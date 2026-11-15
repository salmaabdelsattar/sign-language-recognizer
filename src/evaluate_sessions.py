import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict

from features import load_dataset

df, X, y = load_dataset("data/landmarks_v2.csv")
bad_rows = list(range(100, 200))
keep = ~df.index.isin(bad_rows)
df, X, y = df[keep], X[keep], y[keep]
groups = df["session"].values

print(df.groupby(["session", "label"]).size().unstack(), "\n")

rf = RandomForestClassifier(n_estimators=200, random_state=0)
cv = GroupKFold(n_splits=df["session"].nunique())
pred = cross_val_predict(rf, X, y, groups=groups, cv=cv)
print(f"Leave-one-session-out accuracy: {(pred == y).mean():.3f}\n")

labels = np.unique(y)
print("Confusion matrix (rows = true, cols = predicted):", list(labels))
print(confusion_matrix(y, pred, labels=labels))