import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import load_dataset

SEED = 0  # change to 1, then 2, and rerun to see how much the MLP varies

df, X, y = load_dataset("data/landmarks_v2.csv")

# Drop the first-take recording that was an A shape labeled B (verified visually)
keep = ~df.index.isin(list(range(100, 200)))
df, X, y = df[keep], X[keep], y[keep]
groups = df["session"].values

classes = np.unique(y)
y_idx = np.searchsorted(classes, y)


class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, Xte, epochs=60):
    torch.manual_seed(SEED)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6  # standardize with training stats only
    Xtr_t = torch.tensor((Xtr - mean) / std, dtype=torch.float32)
    Xte_t = torch.tensor((Xte - mean) / std, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    model = MLP(Xtr.shape[1], len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    n = len(Xtr_t)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(model(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(Xte_t).argmax(1).numpy()


def run_sklearn(make_model):
    def run(Xtr, ytr, Xte):
        return make_model().fit(Xtr, ytr).predict(Xte)
    return run


runners = {
    "Logistic Regression": run_sklearn(
        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    "SVM (RBF)": run_sklearn(lambda: make_pipeline(StandardScaler(), SVC())),
    "Random Forest": run_sklearn(lambda: RandomForestClassifier(n_estimators=200, random_state=SEED)),
    "MLP (PyTorch)": train_mlp,
}

sessions = sorted(np.unique(groups))
print(f"Held-out session:    " + "  ".join(f"{s:>5}" for s in sessions) + "   | mean   time")

results = []
for name, run in runners.items():
    t0 = time.time()
    accs = []
    for tr, te in LeaveOneGroupOut().split(X, y_idx, groups):
        pred = run(X[tr], y_idx[tr], X[te])
        accs.append((pred == y_idx[te]).mean())
    secs = time.time() - t0
    results.append((name, accs, secs))
    print(f"{name:20s} " + "  ".join(f"{a:.3f}" for a in accs) + f"   | {np.mean(accs):.3f}  {secs:4.0f}s")

print("\nMarkdown table for your README:\n")
print("| Model | " + " | ".join(f"Session {s}" for s in sessions) + " | Mean |")
print("|---|" + "---|" * (len(sessions) + 1))
for name, accs, _ in results:
    print(f"| {name} | " + " | ".join(f"{a:.3f}" for a in accs) + f" | {np.mean(accs):.3f} |")