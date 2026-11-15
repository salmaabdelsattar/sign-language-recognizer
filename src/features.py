import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

COORD_COLS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]


def normalize(row_63):
    """63 raw landmark values -> 210 pairwise distances.
    Unaffected by hand position, tilt, size in frame, and left/right hand."""
    pts = np.asarray(row_63, dtype=float).reshape(21, 3)
    pts = pts - pts[0]
    pts = pts / (np.linalg.norm(pts[9]) + 1e-6)
    return pdist(pts)


def load_dataset(path="data/landmarks.csv"):
    df = pd.read_csv(path)
    X = np.array([normalize(r) for r in df[COORD_COLS].values])
    return df, X, df["label"].values