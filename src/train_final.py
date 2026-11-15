import joblib
from sklearn.ensemble import RandomForestClassifier

from features import load_dataset

df, X, y = load_dataset("data/landmarks_v2.csv")

# Drop the first-take recording that was an A shape labeled B
keep = ~df.index.isin(list(range(100, 200)))
X, y = X[keep], y[keep]

clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)
joblib.dump(clf, "models/sign_model.joblib")
print(f"Trained on {len(X)} samples, classes: {list(clf.classes_)}")