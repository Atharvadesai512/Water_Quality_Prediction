
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# 1. Load dataset
df = pd.read_csv("data/water_potability.csv")

X = df.drop("Potability", axis=1)
y = df["Potability"]

# 2. Recreate training split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Load trained model
model = joblib.load("models/best_random_forest.joblib")

# 4. Calculate feature importance using training data
result = permutation_importance(
    model,
    X_train,
    y_train,
    scoring="balanced_accuracy",
    n_repeats=10,
    random_state=42,
    n_jobs=1
)

# 5. Create importance table
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": result.importances_mean,
    "Std": result.importances_std
})

importance = importance.sort_values(
    "Importance", ascending=False
)

print("\nFEATURE IMPORTANCE")
print("----------------------------")
print(importance.round(4).to_string(index=False))

# 6. Save results
os.makedirs("reports/figures", exist_ok=True)

importance.to_csv(
    "reports/feature_importance.csv",
    index=False
)

# 7. Plot feature importance
plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Decrease in Balanced Accuracy")
plt.title("Random Forest - Permutation Feature Importance")
plt.tight_layout()

plt.savefig(
    "reports/figures/feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance saved successfully!")
