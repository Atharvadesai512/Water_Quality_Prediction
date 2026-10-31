
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.metrics import confusion_matrix

# 1. Load dataset
df = pd.read_csv("data/water_potability.csv")

X = df.drop("Potability", axis=1)
y = df["Potability"]

# 2. Recreate original training split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Load tuned Random Forest
model = joblib.load("models/best_random_forest.joblib")

# 4. Generate out-of-fold probabilities
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=1
)[:, 1]

# 5. Evaluate different thresholds
results = []

for threshold in np.arange(0.30, 0.91, 0.05):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_train,
        predictions,
        labels=[0, 1]
    ).ravel()

    unsafe_recall = tn / (tn + fp)
    potable_recall = tp / (tp + fn)

    results.append({
        "Threshold": round(float(threshold), 2),
        "Unsafe Recall": unsafe_recall,
        "Potable Recall": potable_recall,
        "False Potable": fp,
        "Balanced Accuracy": (
            unsafe_recall + potable_recall
        ) / 2
    })

# 6. Create results table
results_df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON")
print("------------------------------")
print(results_df.round(4).to_string(index=False))

# 7. Select threshold
# Require at least 95% recall for unsafe water
eligible = results_df[
    results_df["Unsafe Recall"] >= 0.95
]

if eligible.empty:
    raise ValueError(
        "No threshold meets the 95% unsafe recall target."
    )

best = eligible.sort_values(
    ["Potable Recall", "Balanced Accuracy"],
    ascending=False
).iloc[0]

best_threshold = float(best["Threshold"])

print("\nSELECTED THRESHOLD")
print("------------------------------")
print("Threshold:", best_threshold)
print("Unsafe Recall:", round(best["Unsafe Recall"], 4))
print("Potable Recall:", round(best["Potable Recall"], 4))
print("False Potable:", int(best["False Potable"]))

# 8. Save results
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

results_df.to_csv(
    "reports/threshold_comparison.csv",
    index=False
)

with open("models/threshold.json", "w") as file:
    json.dump({
        "threshold": best_threshold,
        "unsafe_recall_target": 0.95
    }, file, indent=4)

# 9. Plot recall comparison
plt.figure(figsize=(9, 5))

plt.plot(
    results_df["Threshold"],
    results_df["Unsafe Recall"],
    marker="o",
    label="Unsafe Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["Potable Recall"],
    marker="o",
    label="Potable Recall"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label="Selected Threshold"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Recall")
plt.title("Classification Threshold Analysis")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "reports/figures/threshold_analysis.png",
    dpi=300
)

plt.show()

print("\nThreshold optimization completed!")
