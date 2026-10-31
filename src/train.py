
import os
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 1. Load dataset
df = pd.read_csv("data/water_potability.csv")

X = df.drop("Potability", axis=1)
y = df["Potability"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=3, random_state=42
    )
}

# 4. Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, classifier in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1": "f1"
        },
        n_jobs=-1
    )

    results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Balanced Accuracy": scores["test_balanced_accuracy"].mean(),
        "F1 Score": scores["test_f1"].mean()
    })

# 5. Create comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Balanced Accuracy", ascending=False
)

print("\nMODEL COMPARISON")
print("-----------------------------")
print(results_df.round(4).to_string(index=False))

# 6. Save results
os.makedirs("reports", exist_ok=True)

results_df.to_csv("reports/model_comparison.csv", index=False)

print("\nResults saved to reports/model_comparison.csv")
