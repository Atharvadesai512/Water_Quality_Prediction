
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv("data/water_potability.csv")

X = df.drop("Potability", axis=1)
y = df["Potability"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=1
    ))
])

# 4. Define hyperparameters
parameters = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [5, 10, 15, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__class_weight": [None, "balanced"]
}

# 5. Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 6. Hyperparameter tuning
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameters,
    n_iter=20,
    scoring="balanced_accuracy",
    cv=cv,
    random_state=42,
    n_jobs=1,
    verbose=1,
    error_score="raise"
)

# 7. Train model
search.fit(X_train, y_train)

# 8. Check tuning results
assert np.isfinite(
    search.cv_results_["mean_test_score"]
).all(), "Some tuning configurations failed."

# 9. Display results
print("\nBest Parameters:")
print(search.best_params_)

print("\nBest Cross-Validation Balanced Accuracy:")
print(round(search.best_score_, 4))

# 10. Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 11. Save best model
joblib.dump(
    search.best_estimator_,
    "models/best_random_forest.joblib"
)

print("\nBest model saved successfully!")

# 12. Save tuning results
results = pd.DataFrame(search.cv_results_)

results.to_csv(
    "reports/tuning_results.csv",
    index=False
)

print("Tuning results saved successfully!")
