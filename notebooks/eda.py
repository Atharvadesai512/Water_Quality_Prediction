import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

df = pd.read_csv("data/water_potability.csv")

X = df.drop(columns=["Potability"])
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_df = X_train.copy()
train_df["Potability"] = y_train

os.makedirs("reports/figures", exist_ok=True)

print("Training data shape:", train_df.shape)

summary = train_df.describe().T

print("\nTraining data summary:")
print(
    summary[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ].round(2)
)

missing = pd.DataFrame({
    "missing_count": X_train.isna().sum(),
    "missing_percentage": X_train.isna().mean() * 100
})

missing = missing.sort_values(
    "missing_percentage",
    ascending=False
)

print("\nMissing values in training data:")
print(missing.round(2))

plt.figure(figsize=(10, 5))

sns.barplot(
    data=missing.reset_index(),
    x="missing_percentage",
    y="index",
    color="steelblue"
)

plt.title("Missing Values in Training Data")
plt.xlabel("Missing values (%)")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    "reports/figures/missing_values.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

feature_columns = X_train.columns

train_df[feature_columns].hist(
    figsize=(15, 12),
    bins=30,
    edgecolor="black"
)

plt.suptitle(
    "Distribution of Water Quality Features",
    fontsize=16,
    y=1.02
)

plt.tight_layout()

plt.savefig(
    "reports/figures/feature_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
fig, axes = plt.subplots(
    3,
    3,
    figsize=(16, 13)
)

axes = axes.flatten()

for index, feature in enumerate(feature_columns):
    sns.boxplot(
        data=train_df,
        x="Potability",
        y=feature,
        ax=axes[index]
    )

    axes[index].set_title(f"{feature} by Potability")
    axes[index].set_xlabel("Potability: 0 = No, 1 = Yes")

plt.tight_layout()

plt.savefig(
    "reports/figures/features_by_potability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

class_medians = (
    train_df
    .groupby("Potability")[feature_columns]
    .median()
    .T
)

class_medians.columns = [
    "Not Potable Median",
    "Potable Median"
]

class_medians["Difference"] = (
    class_medians["Potable Median"]
    - class_medians["Not Potable Median"]
)

print("\nFeature medians by class:")
print(class_medians.round(2))

correlation_matrix = train_df.corr(numeric_only=True)

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Training Data Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    "reports/figures/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()