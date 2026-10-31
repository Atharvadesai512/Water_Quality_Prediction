import pandas as pd

df = pd.read_csv(r"E:\water_quality\water-quality-prediction-ml\data\water_potability.csv")

print(df.head())

print("Rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())


missing_summary = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": df.isna().mean() * 100
})

print(missing_summary.round(2))

print("Number of samples in each class:")
print(df["Potability"].value_counts())

print("\nPercentage in each class:")
print(
    df["Potability"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

majority_accuracy = (
    df["Potability"]
    .value_counts(normalize=True)
    .max()
)

print(f"Always predict the majority class: {majority_accuracy:.2%}")

print("Exact duplicate rows:", df.duplicated().sum())