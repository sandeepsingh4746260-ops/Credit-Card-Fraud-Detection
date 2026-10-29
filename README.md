# Credit Card Fraud Detection using Machine Learning

A machine learning project that compares a **Decision Tree** model with a **Random Forest** model to classify credit card transactions as legitimate or fraudulent.

## Project Overview

This is a binary classification problem:
- `0` — Legitimate transaction
- `1` — Fraudulent transaction

### Workflow
1. Import libraries
2. Load the dataset
3. Preprocess the data
4. Select input features
5. Split the data into training and testing sets
6. Train and evaluate a Decision Tree
7. Train and evaluate a Random Forest
8. Compare the models
9. Make a sample prediction

## Models Used

### Decision Tree Classifier
Learns decision rules from data. It is relatively easy to interpret, but a single tree can overfit and produce false-positive predictions.

### Random Forest Classifier
Combines predictions from multiple decision trees. It often generalizes better than a single tree, but the best model depends on the metric and the cost of mistakes.

## Technologies and Libraries

- Python
- Pandas — data loading and manipulation
- NumPy — numerical operations
- Matplotlib and Seaborn — visualizations
- Scikit-learn — preprocessing, splitting, training and evaluation

## Dataset

Dataset: [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Download `creditcard.csv` and place it in the same directory as the Jupyter Notebook. The target column is `Class`.

The dataset is highly imbalanced: legitimate transactions greatly outnumber fraudulent transactions.

> The dataset is not included in this project. Follow Kaggle's download and access requirements.

## Project Structure

```text
Credit-Card-Fraud-Detection/
├── Credit_Fraud_DecisionTree_then_RandomForest.ipynb
├── creditcard.csv  # Download separately from Kaggle
└── README.md
```

## How to Run

1. Install Python and Visual Studio Code.
2. Install the **Python** and **Jupyter** extensions in VS Code.
3. Download `creditcard.csv` from Kaggle.
4. Place the dataset and notebook in the same folder.
5. Open the notebook in VS Code and select a Python kernel.
6. Run the notebook cells from top to bottom.

The notebook contains a dependency installation cell for the required libraries.

## Data Preprocessing

- Removed duplicate rows.
- Separated predictor features from the target column `Class`.
- Checked for missing values.
- Used `SimpleImputer(strategy="median")` for numerical missing values. The imputer is fitted on the training data only, then applied to the test data.
- Used an 80/20 train-test split with `stratify=y` to preserve the class ratio.

The notebook uses the available predictor columns and excludes `Class` from the inputs to prevent target leakage.

## Evaluation Metrics

Accuracy alone is not sufficient for an imbalanced fraud dataset.

- **Accuracy:** Fraction of all predictions that are correct.
- **Precision:** Of transactions predicted as fraud, the fraction that are truly fraud.
- **Recall:** Of all actual fraud transactions, the fraction detected.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model ranks positive cases above negative cases across thresholds.

## Results

Results observed in the current notebook run:

| Metric | Decision Tree | Random Forest |
|---|---:|---:|
| Accuracy | 99.1471% | 99.9489% |
| Precision | 0.1405 | 0.9853 |
| Recall | 0.8000 | 0.7053 |
| F1-score | 0.2390 | 0.8221 |
| ROC-AUC | 0.9074 | 0.9292 |

### Interpretation

- Random Forest achieved higher accuracy, precision, F1-score and ROC-AUC in this run.
- Decision Tree achieved higher recall, detecting a larger share of the fraud cases.
- Random Forest's higher precision means fewer flagged transactions were false alarms in this test run.
- Random Forest's lower recall means it missed a larger share of actual fraud cases than Decision Tree did.

This is a trade-off. Choose a model based on the cost of missed fraud versus false alarms, not accuracy alone.

**Reproducibility note:** Results can vary with dataset version, preprocessing, library versions and split settings. Run the notebook before quoting metrics in a report or interview.

## Limitations and Future Improvements

- The dataset is highly imbalanced; accuracy can be misleading.
- A time-based split may better represent predicting future transactions than a random split.
- Further work could include precision-recall curves and PR-AUC, threshold tuning on a separate validation set, cross-validation, and cost-sensitive evaluation.
- The current feature-selection step keeps the available predictor columns; it does not perform statistical feature ranking or automated feature selection.
- This is a learning and model-comparison project, not a production-ready fraud detection system.

## Interview Summary

> I built a credit card fraud detection classifier using Python and Scikit-learn. I removed duplicate rows, handled missing numerical values with median imputation, and used an 80/20 stratified train-test split. I trained a Decision Tree as a baseline and compared it with a Random Forest. I evaluated both using accuracy, precision, recall, F1-score and ROC-AUC. In my run, Random Forest improved precision and F1-score, while Decision Tree achieved higher recall. This showed the trade-off between reducing false alarms and detecting as many fraud cases as possible.

## Author

Add your name and GitHub profile before publishing the repository.
