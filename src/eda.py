from sklearn.datasets import load_iris
import pandas as pd

# 1. Load Iris dataset
iris = load_iris()

# 2. Convert dataset into DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# 3. Add target column
df["target"] = iris.target

# 4. Add species names
df["species"] = df["target"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# 5. Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# 6. Display last 5 rows
print("\nLast 5 Rows:")
print(df.tail())

# 7. Dataset shape
print("\nDataset Shape:")
print(df.shape)

# 8. Column names
print("\nColumn Names:")
print(df.columns)

# 9. Data types
print("\nData Types:")
print(df.dtypes)

# 10. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 11. Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# 12. Species names
print("\nTarget Names:")
print(iris.target_names)

# 13. Species count
print("\nSpecies Count:")
print(df["species"].value_counts())