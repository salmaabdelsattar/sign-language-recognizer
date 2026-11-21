import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from features import load_dataset

POSE_NAMES = ["OPEN palm", "PEACE", "POINT", "CURL"]
DROP_POSES = [3]  # e.g. [3] leaves out CURL; poses are numbered 0-3 as above

df1, X1, y1 = load_dataset("data/landmarks_v2.csv")
keep = ~df1.index.isin(list(range(100, 200)))  # drop the mislabeled first take
df2, X2, y2 = load_dataset("data/landmarks_none.csv")

# Each pose was recorded as 50 consecutive samples, in the order above
df2["pose"] = df2.groupby("session").cumcount() // 50
keep2 = ~df2["pose"].isin(DROP_POSES).values
df2, X2, y2 = df2[keep2], X2[keep2], y2[keep2]

df = pd.concat([df1[keep], df2], ignore_index=True)
X = np.vstack([X1[keep], X2])
y = np.concatenate([y1[keep], y2])
groups = df["session"].values

model = RandomForestClassifier(n_estimators=200, random_state=0)
pred = cross_val_predict(model, X, y, groups=groups, cv=LeaveOneGroupOut())

is_letter = y != "NONE"
print(f"Dropped NONE poses: {[POSE_NAMES[p] for p in DROP_POSES] or 'none'}")
print(f"Overall accuracy:               {(pred == y).mean():.3f}")
print(f"Real letters correct:           {(pred[is_letter] == y[is_letter]).mean():.3f}")
print(f"Real letters rejected as NONE:  {(pred[is_letter] == 'NONE').mean():.3f}")
print(f"NONE recognized as NONE:        {(pred[~is_letter] == 'NONE').mean():.3f}\n")

labels = np.unique(y)
print("Labels:", list(labels))
print(confusion_matrix(y, pred, labels=labels))

print("\nPer NONE pose:")
for p, name in enumerate(POSE_NAMES):
    m = (df["pose"] == p).values
    if m.any():
        wrong = pd.Series(pred[m][pred[m] != "NONE"]).value_counts().to_dict()
        print(f"  {name:10s} recognized as NONE: {(pred[m] == 'NONE').mean():.3f}   mistaken for: {wrong}")