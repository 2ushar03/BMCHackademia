# BMCHackademia

## Customer Transaction Analysis

This project performs an extensive analysis of customer transactions, integrating multiple datasets to perform feature engineering and train machine learning models for classification tasks.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Setup and Installation](#setup-and-installation)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Cross-Validation and Overfitting Detection](#cross-validation-and-overfitting-detection)
- [Results](#results)
- [Contributors](#contributors)

## Project Overview

The goal of this project is to analyze customer transaction data, engineer useful features, and build multiple classification models to predict the likelihood of a printer being kept (not returned). We explore various machine learning models and evaluate their performance.

## Dataset Information

This project uses the following datasets:

1. **customer_transaction_info.csv** - Contains customer orders and shipping data.
2. **product_info.csv** - Information about products.
3. **customers_info.csv** - Details of customer demographics.
4. **orders_returned_info.csv** - Information about returned orders.

## Setup and Installation

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd customer-transaction-analysis
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure additional libraries are installed:

   ```bash
   pip install xgboost lightgbm
   ```

## Data Preprocessing

1. **Date Conversion:**
   - Converts Unix timestamps in Order Date and Ship Date columns.

2. **Merging Datasets:**
   - Combines transaction, product, customer, and returns data using appropriate keys.

3. **Handling Missing Values:**
   - Fills missing values in the `Returned` column with `0`.

4. **Encoding Features:**
   - Converts categorical variables using `pd.get_dummies`.
   - Uses `LabelEncoder` for categorical labels.

5. **Scaling Numerical Data:**
   - Standardizes `Sales`, `Quantity`, and `Profit` using `StandardScaler`.

## Feature Engineering

- Creates a boolean flag `Is_Printer_Product` for products categorized as printers.
- Extracts temporal features (`year`, `month`, `day`, `weekday`) from date columns.
- Computes `Profit_Margin` as `Profit / Sales`.
- Introduces `Printer_Kept` as the target variable based on return status.

## Model Training and Evaluation

We implement and evaluate the following models:

1. **Random Forest Classifier**
2. **Logistic Regression**
3. **K-Nearest Neighbors (KNN)**
4. **Support Vector Machine (SVM)**
5. **Gradient Boosting Classifier**
6. **AdaBoost Classifier**
7. **Naive Bayes (GaussianNB)**
8. **XGBoost Classifier**
9. **LightGBM Classifier**
10. **Decision Tree Classifier**

### Performance Metrics

- **Accuracy**: Measures overall correctness.
- **Confusion Matrix**: Evaluates true positives, false positives, etc.
- **Classification Report**: Includes precision, recall, and F1-score.

## Hyperparameter Tuning

Utilizes `GridSearchCV` to optimize hyperparameters of the Random Forest model:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
```

## Cross-Validation and Overfitting Detection

- 5-fold cross-validation is used to detect overfitting.
- Out-of-Bag (OOB) evaluation is conducted on the Random Forest model.

## Results

- Displays accuracy, confusion matrix, and classification report for all models.
- Identifies the best-performing model using hyperparameter tuning and cross-validation.
