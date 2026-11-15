import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import COORD_COLS

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

df = pd.read_csv("data/landmarks_v2.csv")
raw = df[COORD_COLS].values
y = df["label"].values


def last_rows(letter):
    return np.where(y == letter)[0][-100:]  # final recording of that letter


panels = [
    ("rows 100-199 (labeled B)", np.arange(100, 200)),
    ("reference B", last_rows("B")),
    ("reference A", last_rows("A")),
    ("rows 1700-1799 (labeled A)", np.arange(1700, 1800)),
]


def draw(ax, row, title):
    pts = np.asarray(row, dtype=float).reshape(21, 3)
    for a, b in CONNECTIONS:
        ax.plot([pts[a, 0], pts[b, 0]], [-pts[a, 1], -pts[b, 1]], "k-", lw=1.5)
    ax.scatter(pts[:, 0], -pts[:, 1], c="r", s=12)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8)


fig, axes = plt.subplots(len(panels), 3, figsize=(9, 3 * len(panels)))
for r, (name, idx) in enumerate(panels):
    for c, k in enumerate([10, 50, 90]):
        draw(axes[r, c], raw[idx[k]], f"{name}\nframe {k}")
plt.tight_layout()
plt.savefig("data/block_check.png", dpi=120)
print("Saved data/block_check.png")