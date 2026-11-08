# Statlog (German Credit Data) – Credit Risk Classification

## A. Problem Statement

The objective of this project is to develop and evaluate machine learning classification models to predict whether a person represents a good or bad credit risk based on their financial and personal information.

Credit risk classification helps financial institutions assess the potential risk associated with lending money to applicants. In this project, five machine learning algorithms are used and compared in-order to identify their classification performance.

The models that were evaluated are:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

The models are compared for Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

## B. Dataset Description

**Dataset Name:** Statlog (German Credit Data)  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 144  
**Number of Instances:** 1,000  
**Number of Features:** 20  
**Problem Type:** Binary Classification  
**Target Variable:** `class`

The dataset contains information about credit applicants and is used to classify them into two categories:

- **Class 1:** Good credit risk
- **Class 2:** Bad credit risk

The 20 input features include information such as checking account status, credit history, loan purpose, credit amount, savings, employment status, age, housing, and other applicant characteristics.

The original dataset contains 7 numerical and 13 categorical features. It does not have any missing values.

**Dataset Link:**  
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

**Dataset Citation:**

Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

### Important Dataset Consideration

The UCI documentation provides a cost matrix for this dataset:

| Actual Class | Predicted Good (1) | Predicted Bad (2) |
|---|---:|---:|
| Good (1) | 0 | 1 |
| Bad (2) | 5 | 0 |

This indicates that classifying a bad credit applicant as good carries a higher cost than classifying a good credit applicant as bad. Therefore, identifying bad credit applicants is very important.

## C. GitHub Repository Link

The project repository contains the Jupyter Notebook, Streamlit application, test dataset, model comparison results, and saved machine learning models.

**GitHub Repository:**  
https://github.com/ahaz-off4687/statlog-german-credit-classification

### Project Files

| File / Folder | Description |
|---|---|
| `ML ASSGN 2.ipynb` | Jupyter Notebook containing the implementation and evaluation of the machine learning models |
| `app.py` | Streamlit application source code |
| `requirements.txt` | Python dependencies required to run the application |
| `test_data.csv` | Test dataset used by the application |
| `model_comparison.csv` | Performance comparison of the evaluated models |
| `saved_models/` | Folder containing the saved machine learning models |
| `README.md` | Project documentation |

### Saved Models

The following trained models are included in the `saved_models/` folder:

1. `logistic_regression.pkl`
2. `decision_tree.pkl`
3. `knn.pkl`
4. `gaussian_naive_bayes.pkl`
5. `svm.pkl`

## D. Models Used and Performance Comparison

Five machine learning classification algorithms were implemented and evaluated on the test dataset.

The following metrics were used for performance comparison:

- **Accuracy:** Measures the proportion of correctly classified observations.
- **AUC:** Measures the model's ability to distinguish between the two classes.
- **Precision:** Measures how many applicants predicted as bad credit risks actually belong to Class 2.
- **Recall:** Measures how many actual bad credit applicants are correctly identified.
- **F1-score:** Harmonic mean of Precision and Recall.
- **MCC:** Measures the overall quality of binary classifications, considering all four confusion-matrix outcomes.

### Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.780 | 0.8040 | 0.6667 | 0.5333 | 0.5926 | 0.4496 |
| Decision Tree | 0.625 | 0.5655 | 0.3846 | 0.4167 | 0.4000 | 0.1281 |
| KNN | 0.725 | 0.7295 | 0.5676 | 0.3500 | 0.4330 | 0.2782 |
| Gaussian Naive Bayes | 0.695 | 0.7411 | 0.4937 | 0.6500 | 0.5612 | 0.3415 |
| SVM | 0.790 | 0.7960 | 0.7250 | 0.4833 | 0.5800 | 0.4637 |

*Note: Precision, Recall, and F1-score are reported for Class 2 (Bad credit risk).*

### Observations

**1. Logistic Regression**

- Achieved an accuracy of 78%.
- Recorded the highest AUC (0.8040) among the five models.
- Achieved the highest F1-score (0.5926).
- Identified 53.33% of actual bad credit applicants.

**2. Decision Tree**

- Achieved an accuracy of 62.5%.
- Recorded the lowest AUC (0.5655) and MCC (0.1281).
- Showed comparatively weaker classification performance among the evaluated models.

**3. K-Nearest Neighbors (KNN)**

- Achieved an accuracy of 72.5%.
- Recorded a recall of 35% for bad credit applicants.
- Its ability to identify actual bad credit applicants was comparatively limited.

**4. Gaussian Naive Bayes**

- Achieved an accuracy of 69.5%.
- Recorded the highest recall (65%) for bad credit applicants.
- Identified a greater proportion of actual bad credit applicants compared with the other models, although its precision was relatively low.

**5. Support Vector Machine (SVM)**

- Achieved the highest accuracy of 79%.
- Recorded the highest MCC (0.4637).
- Achieved the highest precision (0.7250) for bad credit applicants.
- Demonstrated the strongest overall performance based on accuracy and MCC among the evaluated models.

### Overall Conclusion

Based on the evaluated metrics, Support Vector Machine (SVM) achieved the highest accuracy (79%) and MCC (0.4637), making it the best-performing model according to these measures.

Logistic Regression achieved the highest AUC and F1-score, while Gaussian Naive Bayes achieved the highest recall for Class 2 (Bad credit risk).

Considering the cost matrix provided by the UCI repository, recall for bad credit applicants is also an important evaluation measure. Gaussian Naive Bayes identified the highest proportion of bad credit applicants, whereas SVM provided better precision and overall accuracy.

Therefore, the choice of the final model depends on the relative importance of correctly identifying bad credit applicants and minimizing incorrect classifications. and so I would suggest consider these two models to be implemented simultaneously for better results.

## Streamlit Application

The project is also deployed as a Streamlit web application.

**Live Application:**  
https://south-german-credit-classification.streamlit.app

