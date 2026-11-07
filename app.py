
import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Credit Classification",
    page_icon="📊",
    layout="wide"
)

st.title("South German Credit Classification")
st.write("Compare five trained machine learning models.")

# Available trained models
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.pkl",
    "SVM": "svm.pkl"
}

# Model selection
selected_model = st.selectbox(
    "Select a classification model",
    list(model_files.keys())
)

# CSV upload
uploaded_file = st.file_uploader(
    "Upload your labelled test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(data.head())

    # Check target column
    if "class" not in data.columns:

        st.error("The uploaded CSV must contain a 'class' column.")

    else:

        X_new = data.drop(columns=["class"])
        y_actual = data["class"]

        # Check model file
        model_path = os.path.join(
            "saved_models",
            model_files[selected_model]
        )

        if not os.path.exists(model_path):

            st.error("Selected model file was not found.")

        else:

            # Load trained model
            model = joblib.load(model_path)

            # Make predictions
            y_pred = model.predict(X_new)

            # Get prediction probabilities
            y_prob = model.predict_proba(X_new)

            # Find probability column for class 2
            class_index = list(model.classes_).index(2)

            auc = roc_auc_score(
                (y_actual == 2).astype(int),
                y_prob[:, class_index]
            )

            # Calculate performance metrics
            accuracy = accuracy_score(y_actual, y_pred)

            precision = precision_score(
                y_actual, y_pred,
                pos_label=2,
                zero_division=0
            )

            recall = recall_score(
                y_actual, y_pred,
                pos_label=2,
                zero_division=0
            )

            f1 = f1_score(
                y_actual, y_pred,
                pos_label=2,
                zero_division=0
            )

            mcc = matthews_corrcoef(y_actual, y_pred)

            # Display metrics
            st.subheader("Model Performance")

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", round(accuracy, 4))
            col2.metric("AUC", round(auc, 4))
            col3.metric("Precision", round(precision, 4))

            col4, col5, col6 = st.columns(3)

            col4.metric("Recall", round(recall, 4))
            col5.metric("F1 Score", round(f1, 4))
            col6.metric("MCC", round(mcc, 4))

            # Confusion matrix
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_actual,
                y_pred,
                labels=[1, 2]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual 1", "Actual 2"],
                columns=["Predicted 1", "Predicted 2"]
            )

            st.dataframe(cm_df)

            # Classification report
            st.subheader("Classification Report")

            report = classification_report(
                y_actual,
                y_pred,
                labels=[1, 2],
                zero_division=0,
                output_dict=True
            )

            st.dataframe(pd.DataFrame(report).transpose())

            # Display predictions
            st.subheader("Prediction Results")

            result_data = data.copy()
            result_data["Predicted Class"] = y_pred

            st.dataframe(result_data)
