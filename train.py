"""
train.py  —  Step 2
Full alphabet A-Z + SPACE + DELETE
Reads landmark coordinates from CSV — no image re-detection needed
"""

import os
import csv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

CSV_FILE   = "landmarks.csv"
MODEL_FILE = "model.pkl"

if not os.path.exists(CSV_FILE):
    print(f"ERROR: {CSV_FILE} not found. Run  python collect_data.py  first.")
    exit()

print("\n===  Training Sign Language Model  ===")
print(f"Reading from {CSV_FILE} ...\n")

X, y = [], []
with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) == 64:
            y.append(row[0])
            X.append([float(v) for v in row[1:]])

print(f"Total samples: {len(X)}")
counts = Counter(y)
for letter, cnt in sorted(counts.items()):
    bar = "█" * (cnt // 10)
    print(f"  {letter:7s}: {cnt:4d}  {bar}")
print()

if len(X) < 100:
    print("ERROR: Not enough data. Collect more samples first.")
    exit()

X     = np.array(X)
le    = LabelEncoder()
y_enc = le.fit_transform(np.array(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print("Training Random Forest (300 trees) ...")
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(f"\nTest accuracy: {acc:.2%}")

if acc >= 0.95:
    print("  Excellent!")
elif acc >= 0.85:
    print("  Good! App should work well.")
elif acc >= 0.70:
    print("  Okay — consider recollecting some letters for better results.")
else:
    print("  Low accuracy — recollect data with better lighting and hand positioning.")

with open(MODEL_FILE, "wb") as f:
    pickle.dump({"model": clf, "label_encoder": le}, f)

print(f"\nModel saved to {MODEL_FILE}")
print("Labels:", list(le.classes_))
print("\nNext:  python app.py\n")
