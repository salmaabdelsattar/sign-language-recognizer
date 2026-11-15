import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split

from features import COORD_COLS

df = pd.read_csv("data/landmarks.csv")
y = df["label"].values
raw = df[COORD_COLS].values


def base(row):
    pts = np.asarray(row, dtype=float).reshape(21, 3)
    pts = pts - pts[0]                         # wrist = origin
    return pts / (np.linalg.norm(pts[9]) + 1e-6)  # scale by palm size


def feat_basic(row):
    return base(row).flatten()


def feat_rotated(row):
    pts = base(row)
    # rotate in the image plane so wrist -> middle-finger base points straight up
    d = -np.pi / 2 - np.arctan2(pts[9, 1], pts[9, 0])
    R = np.array([[np.cos(d), -np.sin(d)], [np.sin(d), np.cos(d)]])
    pts[:, :2] = pts[:, :2] @ R.T
    return pts.flatten()


def feat_dist(row):
    return pdist(base(row))  # 210 pairwise distances


# Split each letter's recording into 5 consecutive blocks.
# Block CV trains on 4 blocks and tests on the held-out one, 5 times.
groups = np.zeros(len(df), dtype=int)
for label in np.unique(y):
    idx = np.where(y == label)[0]
    groups[idx] = np.arange(len(idx)) * 5 // len(idx)

feature_sets = {
    "basic (no mirroring)": feat_basic,
    "rotation-normalized": feat_rotated,
    "pairwise distances": feat_dist,
}

for name, fn in feature_sets.items():
    X = np.array([fn(r) for r in raw])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    random_acc = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    block_acc = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=0),
        X, y, groups=groups, cv=GroupKFold(n_splits=5),
    ).mean()
    print(f"{name:22s} random split: {random_acc:.3f} | block CV (honest): {block_acc:.3f}")