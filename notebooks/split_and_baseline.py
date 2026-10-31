import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("data/water_potability.csv")

X = df.drop(columns=["Potability"])
y = df["Potability"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

def show_distribution(name, target):
    counts = target.value_counts().sort_index()
    percentages = (
        target.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    result = pd.DataFrame({
        "count": counts,
        "percentage": percentages
    })

    print(f"\n{name}:")
    print(result)


show_distribution("Complete dataset", y)
show_distribution("Training set", y_train)
show_distribution("Test set", y_test)

dummy_model = DummyClassifier(strategy="most_frequent")

dummy_model.fit(X_train, y_train)

dummy_predictions = dummy_model.predict(X_test)

print("\nDummy model predictions:")
print(pd.Series(dummy_predictions).value_counts())
accuracy = accuracy_score(y_test, dummy_predictions)
balanced_accuracy = balanced_accuracy_score(
    y_test,
    dummy_predictions
)

print(f"\nAccuracy: {accuracy:.2%}")
print(f"Balanced accuracy: {balanced_accuracy:.2%}")

matrix = confusion_matrix(y_test, dummy_predictions)

confusion_table = pd.DataFrame(
    matrix,
    index=["Actual Not Potable", "Actual Potable"],
    columns=["Predicted Not Potable", "Predicted Potable"]
)

print("\nConfusion matrix:")
print(confusion_table)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        dummy_predictions,
        target_names=["Not Potable", "Potable"],
        zero_division=0
    )
)