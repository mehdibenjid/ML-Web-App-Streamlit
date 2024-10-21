"""
Binary Classification Web App using Streamlit.

This application allows users to upload the Mushroom Classification dataset and select between
three classifiers: Support Vector Machine (SVM), Logistic Regression, and Random Forest.
Users can adjust hyperparameters, train the model, and visualize performance metrics such as
Confusion Matrix, ROC Curve, and Precision-Recall Curve.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.metrics import precision_score, recall_score

# Set the maximum line length for Pylint
# pylint: disable=C0301


@st.cache_data
def load_data():
    """
    Load and preprocess the mushroom dataset.

    Returns:
        pd.DataFrame: The preprocessed mushroom dataset.
    """
    data = pd.read_csv("mushrooms.csv")
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


@st.cache_data
def split_data(df):
    """
    Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): The preprocessed mushroom dataset.

    Returns:
        tuple: Training and testing data splits.
    """
    x = df.drop("class", axis=1)
    y = df["class"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0
    )
    return x_train, x_test, y_train, y_test


def plot_metrics(metrics_list, model, x_test, y_test, y_pred):
    """
    Plot selected performance metrics.

    Args:
        metrics_list (list): List of metrics to plot.
        model: Trained machine learning model.
        x_test (pd.DataFrame): Testing features.
        y_test (pd.Series): True labels for testing data.
        y_pred (np.array): Predicted labels for testing data.
    """
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["edible", "poisonous"]
        )
        disp.plot(ax=ax_cm)
        st.pyplot(fig_cm)

    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(x_test)
        else:
            y_score = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        fig_roc, ax_roc = plt.subplots()
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax_roc)
        st.pyplot(fig_roc)

    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(x_test)
        else:
            y_score = model.predict_proba(x_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        fig_pr, ax_pr = plt.subplots()
        PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=ax_pr)
        st.pyplot(fig_pr)


def train_model(classifier_name, x_train, y_train, **kwargs):
    """
    Train the selected classifier with given hyperparameters.

    Args:
        classifier_name (str): Name of the classifier.
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        **kwargs: Hyperparameters for the classifier.

    Returns:
        model: Trained machine learning model.
    """
    if classifier_name == "Support Vector Machine (SVM)":
        model = SVC(**kwargs)
    elif classifier_name == "Logistic Regression":
        model = LogisticRegression(**kwargs)
    elif classifier_name == "Random Forest":
        model = RandomForestClassifier(**kwargs)
    else:
        st.error("Invalid classifier selected.")
        return None

    model.fit(x_train, y_train)
    return model


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")

    df = load_data()
    x_train, x_test, y_train, y_test = split_data(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier",
        ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"),
    )

    metrics = st.sidebar.multiselect(
        "What metrics to plot?",
        ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),
    )

    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        c = st.sidebar.number_input(
            "C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_SVM"
        )
        kernel = st.sidebar.radio(
            "Kernel", ("linear", "poly", "rbf", "sigmoid"), key="kernel"
        )
        gamma = st.sidebar.radio(
            "Gamma (Kernel coefficient)", ("scale", "auto"), key="gamma"
        )
        classifier_kwargs = {
            "C": c,
            "kernel": kernel,
            "gamma": gamma,
            "probability": True,
        }

    elif classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        c = st.sidebar.number_input(
            "C (Inverse of regularization strength)",
            0.01,
            10.0,
            step=0.01,
            key="C_LR",
        )
        max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
        classifier_kwargs = {"C": c, "max_iter": max_iter}

    elif classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "Number of trees", 100, 1000, step=10, key="n_estimators"
        )
        max_depth = st.sidebar.number_input(
            "Max depth", 1, 20, step=1, key="max_depth"
        )
        bootstrap = st.sidebar.radio("Bootstrap samples", (True, False), key="bootstrap")
        classifier_kwargs = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "bootstrap": bootstrap,
            "n_jobs": -1,
        }

    else:
        st.error("Please select a classifier.")
        return

    if st.sidebar.button("Classify", key="classify"):
        st.subheader(f"{classifier} Results")
        model = train_model(classifier, x_train, y_train, **classifier_kwargs)
        if model is not None:
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy:", round(accuracy, 2))
            st.write("Precision:", round(precision_score(y_test, y_pred), 2))
            st.write("Recall:", round(recall_score(y_test, y_pred), 2))
            plot_metrics(metrics, model, x_test, y_test, y_pred)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)


if __name__ == "__main__":
    main()
