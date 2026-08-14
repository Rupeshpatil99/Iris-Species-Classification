# 🌸 Iris Species Classification

A machine learning classification project that predicts the species of an iris flower based on its sepal and petal measurements.

The project demonstrates an end-to-end beginner-friendly Data Science workflow using **Python, Pandas, Scikit-learn, Matplotlib, and Seaborn**.

---

## 📌 Project Overview

The Iris Species Classification project uses machine learning to classify iris flowers into three species:

* **Iris Setosa**
* **Iris Versicolor**
* **Iris Virginica**

The model uses four numerical features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The project includes data analysis, exploratory data analysis (EDA), visualization, model training, prediction, and model evaluation.

---

## 🎯 Project Objectives

1. Perform exploratory data analysis on the Iris dataset using Pandas.
2. Understand relationships between flower measurements and species.
3. Visualize feature distributions and relationships using Matplotlib and Seaborn.
4. Prepare the dataset for machine learning.
5. Build a classification model using Scikit-learn.
6. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Achieve at least **95% classification accuracy**.
8. Compare and optimize different machine learning algorithms.

---

## 🛠️ Technologies Used

| Technology       | Purpose                             |
| ---------------- | ----------------------------------- |
| Python           | Programming language                |
| Pandas           | Data analysis and manipulation      |
| NumPy            | Numerical computing                 |
| Scikit-learn     | Machine learning                    |
| Matplotlib       | Data visualization                  |
| Seaborn          | Statistical visualization           |
| Jupyter Notebook | Data Science experimentation        |
| Git & GitHub     | Version control and project hosting |

---

## 📊 Dataset

The project uses the classic **Iris dataset** provided by Scikit-learn.

### Dataset characteristics

* **150 observations**
* **4 input features**
* **3 target classes**
* **No missing values**
* **50 samples per species**

### Features

| Feature      | Description                |
| ------------ | -------------------------- |
| Sepal Length | Length of the flower sepal |
| Sepal Width  | Width of the flower sepal  |
| Petal Length | Length of the flower petal |
| Petal Width  | Width of the flower petal  |

### Target Classes

| Target | Species    |
| -----: | ---------- |
|      0 | Setosa     |
|      1 | Versicolor |
|      2 | Virginica  |

---

## 🔍 Exploratory Data Analysis

The project performs several EDA operations using Pandas:

* Dataset shape and structure
* Column and data type inspection
* Missing-value detection
* Statistical summary
* Class distribution
* Feature distribution analysis
* Feature correlation analysis

### Visualizations

The project includes:

* Species distribution plot
* Feature histograms
* Box plots
* Scatter plots
* Pair plots
* Correlation heatmap

EDA showed that **petal length and petal width provide strong separation between the iris species**, making them useful features for classification.

---

## 🤖 Machine Learning Workflow

The machine learning pipeline follows these steps:

```text
Iris Dataset
      ↓
Data Loading
      ↓
Data Analysis
      ↓
Exploratory Data Analysis
      ↓
Feature & Target Separation
      ↓
Train-Test Split
      ↓
Model Training
      ↓
Prediction
      ↓
Model Evaluation
      ↓
Model Optimization
```

---

## 🧪 Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

This produces:

```text
Training samples: 120
Testing samples:   30
```

A fixed `random_state=42` is used for reproducibility, and stratified sampling maintains the class distribution.

---

## 🧠 Machine Learning Model

The first classification model implemented is:

### Logistic Regression

The model is trained using:

```python
model.fit(X_train, y_train)
```

Predictions are generated using:

```python
y_pred = model.predict(X_test)
```

---

## 📈 Model Performance

### Logistic Regression Results

The current model achieved:

**96.67% test accuracy**

### Classification Performance

| Class       | Precision |   Recall | F1-Score |
| ----------- | --------: | -------: | -------: |
| Setosa      |      1.00 |     1.00 |     1.00 |
| Versicolor  |      1.00 |     0.90 |     0.95 |
| Virginica   |      0.91 |     1.00 |     0.95 |
| **Overall** |  **0.97** | **0.97** | **0.97** |

### Confusion Matrix

```text
                Predicted
              Setosa  Versicolor  Virginica

Actual Setosa     10        0          0
Actual Versicolor  0        9          1
Actual Virginica   0        0         10
```

The model correctly classified **29 out of 30 test samples**, resulting in **96.67% accuracy**.

---

## 📁 Project Structure

```text
iris-species-classification/
│
├── data/
│
├── model/
│
├── notebooks/
│
├── src/
│   ├── eda.py
│   ├── visualization.py
│   └── train_model.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

### File Description

**`src/eda.py`**

Performs basic data analysis using Pandas.

**`src/visualization.py`**

Creates visualizations for exploratory data analysis.

**`src/train_model.py`**

Prepares the dataset, trains the Logistic Regression model, generates predictions, and evaluates performance.

**`requirements.txt`**

Contains the Python dependencies required to run the project.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rupeshpatil99/-Iris-Species-Classification.git
```

### 2. Navigate to the project

```bash
cd -Iris-Species-Classification
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run Exploratory Data Analysis

```bash
python src/eda.py
```

### Run Visualizations

```bash
python src/visualization.py
```

### Train and evaluate the model

```bash
python src/train_model.py
```

The training script displays:

* Model accuracy
* Classification report
* Confusion matrix

---

## 💡 Example Prediction

The model takes four measurements:

```text
Sepal Length
Sepal Width
Petal Length
Petal Width
```

Example input:

```text
5.1, 3.5, 1.4, 0.2
```

The model can predict:

```text
Iris Setosa
```

---

## 🚀 Future Improvements

The project will be further improved by:

* Comparing multiple classification algorithms
* Hyperparameter tuning
* Cross-validation
* Feature scaling using Scikit-learn pipelines
* Saving the best trained model
* Creating a separate prediction script
* Building a simple web interface
* Deploying the model as an ML application
* Adding model performance visualizations

### Planned Model Comparison

```text
Logistic Regression
        ↓
Decision Tree
        ↓
K-Nearest Neighbors
        ↓
Random Forest
        ↓
Support Vector Machine
        ↓
Compare Performance
        ↓
Select Best Model
        ↓
Hyperparameter Optimization
```

---

## 📚 Key Data Science Concepts Demonstrated

This project demonstrates practical understanding of:

* Data loading
* Data cleaning
* Exploratory Data Analysis
* Data visualization
* Feature selection
* Target variable
* Train-test splitting
* Classification
* Logistic Regression
* Model prediction
* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Model evaluation
* Machine learning workflow

---

## 👨‍💻 Author

**Rupesh Patil**

Bachelor of Engineering — Computer Engineering

GitHub: [Rupeshpatil99](https://github.com/Rupeshpatil99)

---

## 📄 License

This project is created for educational and portfolio purposes.
