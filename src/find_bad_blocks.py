import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

from features import load_dataset

df, X, y = load_dataset("data/landmarks_v2.csv")
groups = df["session"].values

rf = RandomForestClassifier(n_estimators=200, random_state=0)
pred = cross_val_predict(rf, X, y, groups=groups, cv=GroupKFold(n_splits=4))

# Each recording is a run of consecutive rows with the same session + letter
key = df["session"].astype(str) + df["label"]
block = (key != key.shift()).cumsum().values

print("Recordings with accuracy below 90%:\n")
for b in np.unique(block):
    idx = np.where(block == b)[0]
    acc = (pred[idx] == y[idx]).mean()
    if acc < 0.9:
        counts = pd.Series(pred[idx]).value_counts().to_dict()
        print(f"rows {idx[0]}-{idx[-1]} | session {df['session'].iloc[idx[0]]} | "
              f"letter {y[idx[0]]} | accuracy {acc:.2f} | predicted as {counts}")
print("\nDone")