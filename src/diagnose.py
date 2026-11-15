import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from features import COORD_COLS, normalize

df = pd.read_csv("data/landmarks.csv")
y = df["label"].values

print("Handedness label per letter:")
print(df.groupby("label")["hand"].value_counts(), "\n")

for mirror in (True, False):
    X = np.array([
        normalize(r[COORD_COLS].values, r["hand"] if mirror else "Right")
        for _, r in df.iterrows()
    ])

    # Random split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
    print(f"mirroring={mirror} | random split accuracy: {rf.score(Xte, yte):.3f}")

    # Time-ordered split (what train_baseline.py used)
    tr, te = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        cut = int(len(idx) * 0.7)
        tr += list(idx[:cut])
        te += list(idx[cut:])
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X[tr], y[tr])
    print(f"mirroring={mirror} | time split accuracy:   {rf.score(X[te], y[te]):.3f}\n")