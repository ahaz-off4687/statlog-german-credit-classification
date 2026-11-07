
# South German Credit Classification

## 1. Project Overview

This project compares five machine learning classification algorithms using the South German Credit dataset.

The objective is to classify credit applicants into two classes using customer and financial information.

## 2. Dataset Information

- Dataset: South German Credit
- Source: UCI Machine Learning Repository
- Dataset ID: 522
- Total records: 1000
- Input features: 20
- Target variable: class
- Missing values: None
- Number of classes: 2

## 3. Machine Learning Models

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Data Preprocessing

- Categorical features were encoded using One-Hot Encoding.
- Numerical features were scaled using StandardScaler.
- Data was divided into training and testing sets.
- Stratified sampling was used to preserve class proportions.

## 5. Evaluation Metrics

The models were evaluated using:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)

## 6. Model Performance

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.780 | 0.8040 | 0.6667 | 0.5333 | 0.5926 | 0.4496 |
| Decision Tree | 0.625 | 0.5655 | 0.3846 | 0.4167 | 0.4000 | 0.1281 |
| KNN | 0.725 | 0.7295 | 0.5676 | 0.3500 | 0.4330 | 0.2782 |
| Gaussian Naive Bayes | 0.695 | 0.7411 | 0.4937 | 0.6500 | 0.5612 | 0.3415 |
| SVM | 0.790 | 0.7960 | 0.7250 | 0.4833 | 0.5800 | 0.4637 |

## 7. Results and Observations

SVM achieved the highest accuracy of 79% and the highest MCC of 0.4637.

Logistic Regression achieved the highest AUC of 0.8040 and F1-score of 0.5926.

Gaussian Naive Bayes achieved the highest recall of 0.65, identifying more class 2 instances than the other models.

Based on MCC, SVM was selected as the overall best-performing model.

## 8. Streamlit Application

A Streamlit web application was developed to evaluate the saved classification models.

Application features:

- Upload labelled CSV test data.
- Select one of the five trained models.
- Display evaluation metrics.
- Display confusion matrix.
- Display classification report.
- Display prediction results.

The application loads previously trained models and does not retrain them.

## 9. Project Structure

ML Assignment 2/
- app.py
- requirements.txt
- README.md
- test_data.csv
- model_comparison.csv
- saved_models/
- ML ASSGN 2.ipynb

## 10. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## 11. Dataset Reference

UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/522/south+german+credit
