from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

# Add species name
df["species"] = df["target"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print(df.head())
sns.countplot(data=df, x="species")

plt.title("Iris Species Distribution")
plt.xlabel("Species")
plt.ylabel("Count")

plt.show()

df.hist(figsize=(10, 8))

plt.suptitle("Distribution of Iris Features")

plt.show()


df.hist(figsize=(10, 8))

plt.suptitle("Distribution of Iris Features")

plt.show()

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="species",
    y="petal length (cm)"
)

plt.title("Petal Length by Species")

plt.show()
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    s=100
)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")

plt.show()

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    hue="species",
    s=100
)

plt.title("Sepal Length vs Sepal Width")

plt.show()



sns.pairplot(
    df,
    hue="species"
)

plt.show()

features = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

correlation = df[features].corr()

plt.figure(figsize=(10, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.show()