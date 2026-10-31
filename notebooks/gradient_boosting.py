
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# 1. Load dataset
df = pd.read_csv("data/water_potability.csv")

# 2. Separate features and target
X = df.drop("Potability", axis=1)
y = df["Potability"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create Gradient Boosting pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

# 5. Cross-validation on training data
cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

scores = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1"
    }
)

# 6. Display validation results
print("Gradient Boosting - 5 Fold Cross Validation")
print("-------------------------------------------")

print("Mean Accuracy:",
      round(scores["test_accuracy"].mean(), 4))

print("Mean Balanced Accuracy:",
      round(scores["test_balanced_accuracy"].mean(), 4))

print("Mean Potable F1-score:",
      round(scores["test_f1"].mean(), 4))

# 7. Train model on full training data
model.fit(X_train, y_train)

print("\nGradient Boosting training completed!")
print("Test set reserved for final evaluation.")
