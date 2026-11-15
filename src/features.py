import numpy as np
import pandas as pd

COORD_COLS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]


def normalize(row_63, hand="Right"):
    """Turn 63 raw landmark values into position- and size-independent features."""
    pts = np.asarray(row_63, dtype=float).reshape(21, 3)
    pts = pts - pts[0]                      # wrist becomes the origin
    scale = np.linalg.norm(pts[9]) + 1e-6   # wrist to middle-finger base
    pts = pts / scale
    if hand == "Left":                      # mirror so both hands look alike
        pts[:, 0] = -pts[:, 0]
    return pts.flatten()


def load_dataset(path="data/landmarks.csv"):
    df = pd.read_csv(path)
    X = np.array([normalize(r[COORD_COLS].values, r["hand"]) for _, r in df.iterrows()])
    return df, X, df["label"].values