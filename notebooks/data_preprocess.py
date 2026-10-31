
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("data/water_potability.csv")

# 2. Separate input and output
X = df.drop("Potability", axis=1)
y = df["Potability"]

# 3. Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Handle missing values
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 5. Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Verify preprocessing
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("Missing values in training:",
      pd.DataFrame(X_train).isnull().sum().sum())

print("Missing values in testing:",
      pd.DataFrame(X_test).isnull().sum().sum())

print("Preprocessing completed successfully!")

df = pd.read_csv("data/water_potability.csv")

# 2. Separate input and output
X = df.drop("Potability", axis=1)
y = df["Potability"]

# 3. Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Handle missing values
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 5. Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Verify preprocessing
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("Missing values in training:",
      pd.DataFrame(X_train).isnull().sum().sum())

print("Missing values in testing:",
      pd.DataFrame(X_test).isnull().sum().sum())

print("Preprocessing completed successfully!")
