
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)

# 1. Load dataset
df = pd.read_csv("data/water_potability.csv")

X = df.drop("Potability", axis=1)
y = df["Potability"]

# 2. Recreate the original test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Load tuned model
model = joblib.load("models/best_random_forest.joblib")

# 4. Predict on test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 5. Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nFINAL RANDOM FOREST EVALUATION")
print("--------------------------------")

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Potable", "Potable"],
    zero_division=0
))

# 6. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

print("\nUnsafe water predicted as potable:", cm[0, 1])

# 7. Create report folder
os.makedirs("reports/figures", exist_ok=True)

# 8. Plot confusion matrix
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Potable", "Potable"]
).plot(cmap="Blues")

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(
    "reports/figures/final_confusion_matrix.png",
    dpi=300
)
plt.show()

# 9. Plot ROC curve
RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    name="Random Forest"
)

plt.title("Random Forest ROC Curve")
plt.tight_layout()
plt.savefig(
    "reports/figures/roc_curve.png",
    dpi=300
)
plt.show()

print("\nEvaluation completed successfully!")
