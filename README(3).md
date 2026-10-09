# 🏠 House Price Prediction

A machine learning project that predicts house prices from key property characteristics using **Python, Pandas, Seaborn, Matplotlib, and Scikit-learn**.

This repository demonstrates an end-to-end beginner-to-intermediate machine learning workflow: data loading, exploratory data analysis (EDA), data cleaning, feature selection, model training, evaluation, and prediction on a new house.

---

## 📌 Project Overview

The objective of this project is to build a regression model that estimates the price of a house based on:

- **Area**
- **Bedrooms**
- **Bathrooms**
- **Stories**
- **Parking**

The original dataset contains **545 house records and 13 columns**. For the current model, the analysis focuses on six columns: five predictors plus the target variable, `price`.

### Target Variable

`price`

### Input Features

`area`, `bedrooms`, `bathrooms`, `stories`, `parking`

---

## 🎯 Business Problem

House prices depend on several property characteristics. A predictive model can help provide a data-driven estimate of a property's market value from a small set of measurable attributes.

This project is a practical demonstration of how regression techniques can be applied to a real-world pricing problem.

---

## 🧠 Machine Learning Approach

The project follows this workflow:

```text
Raw Dataset
    ↓
Data Loading
    ↓
Exploratory Data Analysis
    ↓
Feature Selection
    ↓
Duplicate Check / Data Cleaning
    ↓
Train-Test Split
    ↓
Linear Regression
    ↓
Prediction
    ↓
Model Evaluation
```

### 1. Data Loading

The dataset is loaded from:

```python
df = pd.read_csv('Housing.csv')
```

### 2. Exploratory Data Analysis

The notebook performs:

- Dataset shape inspection
- Preview of records
- Data type inspection
- Descriptive statistics
- Missing-value checks
- Duplicate detection
- Correlation analysis
- Histograms
- Count plots
- Box plots
- Correlation heatmap

The correlation analysis shows that **area** has the strongest linear correlation with price among the selected numerical features, followed closely by **bathrooms**.

### 3. Data Cleaning and Feature Selection

The working dataset is reduced to:

```python
['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
```

Duplicate rows are removed before modeling.

The notebook found:

- **545 original rows**
- **1 duplicate row**

No missing values were present in the dataset.

### 4. Train-Test Split

The data is split into:

- **80% training data**
- **20% testing data**
- `random_state = 42`

### 5. Model

The current implementation uses:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
```

Linear Regression was selected as the baseline regression model for this project.

---

## 📊 Model Performance

The model was evaluated using standard regression metrics.

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 1,141,982.03 |
| Mean Squared Error (MSE) | 2,295,578,854,720.57 |
| Root Mean Squared Error (RMSE) | 1,515,116.78 |
| R² Score | 0.5336 |

### Interpretation

The current model explains approximately **53.36% of the variance in house prices** on the test set.

This makes the model a useful **baseline**, but there is substantial room for improvement. The current notebook excludes several categorical variables available in the original dataset, so additional feature engineering and model comparison could potentially improve predictive performance.

> **Note:** The reported metrics are taken from the current notebook implementation and should not be interpreted as a production-grade valuation model.

---

## 🔎 Example Prediction

The notebook tests the model on a new house with:

```python
area = 2500
bedrooms = 4
bathrooms = 3
stories = 2
parking = 2
```

The resulting predicted price is approximately:

```text
6,764,295.70
```

This prediction is generated directly from the trained Linear Regression model.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical operations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Model training, splitting, and evaluation |
| Jupyter Notebook | Development and experimentation |

---

## 📂 Project Structure

A recommended repository structure is:

```text
House-Price-Prediction/
│
├── House_price.ipynb
├── Housing.csv
├── README.md
└── requirements.txt
```

```

## 📈 Key Insights from the Analysis

Based on the current notebook:

- **Area** shows the strongest correlation with house price among the selected features.
- **Bathrooms** also show a relatively strong positive relationship with price.
- **Stories, parking, and bedrooms** have positive but weaker correlations with price.
- The dataset contains **545 records with no missing values**.
- The current model provides a reasonable baseline, but the R² score indicates that house price variation is not fully captured by the selected features.

---

## 👩‍💻 Author

**Lipsa Mahakul**

Aspiring Data Analyst / Data Scientist focused on building practical machine learning and analytics projects.

---

## ⭐ Acknowledgements

This project uses a housing dataset containing property attributes and prices for regression analysis.

---
