import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, 
    log_loss, fbeta_score, jaccard_score, hamming_loss
)
from fpdf import FPDF
from io import BytesIO
import base64
import os

# Load CSS file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("styles.css")

# Function to convert image to base64
def img_to_base64(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Add logo with custom CSS
logo_base64 = img_to_base64("Logo.PNG")
st.markdown(
    f"""
    <style>
    .logo-container {{
        display: flex;
        justify-content: flex-start;
        align-items: center;
        position: fixed;
        bottom: 50px;
        right: 10px;
    }}
    .logo-container img {{
        width: 100px;  /* Adjust the width as needed */
    }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# Define parameters
general_parameters = [
    'Heart Rate', 'Blood Pressure Systolic', 'Blood Pressure Diastolic', 'Respiratory Rate', 'Oxygen Saturation',
    'Temperature', 'Weight', 'Height', 'BMI', 'Blood Glucose', 'Cholesterol', 'HDL', 'LDL', 'Triglycerides',
    'Hemoglobin', 'Hematocrit', 'WBC Count', 'RBC Count', 'Platelet Count', 'Creatinine', 'BUN', 'Sodium',
    'Potassium', 'Calcium', 'Magnesium'
]

als_specific_parameters = [
    'Muscle Strength', 'Motor Function Score', 'Speech Clarity', 'Swallowing Function', 'Respiratory Capacity'
]

parameters = general_parameters + als_specific_parameters

# Function to create realistic fake data
@st.cache_data
def create_realistic_data(num_patients=1000):
    np.random.seed(0)
    data = np.column_stack([
        np.random.normal(70, 10, num_patients),        # Heart Rate
        np.random.normal(120, 15, num_patients),       # Blood Pressure Systolic
        np.random.normal(80, 10, num_patients),        # Blood Pressure Diastolic
        np.random.normal(16, 2, num_patients),         # Respiratory Rate
        np.random.normal(98, 2, num_patients),         # Oxygen Saturation
        np.random.normal(36.6, 0.5, num_patients),     # Temperature
        np.random.normal(70, 15, num_patients),        # Weight
        np.random.normal(1.7, 0.1, num_patients),      # Height
        np.random.normal(25, 5, num_patients),         # BMI
        np.random.normal(100, 15, num_patients),       # Blood Glucose
        np.random.normal(200, 30, num_patients),       # Cholesterol
        np.random.normal(50, 10, num_patients),        # HDL
        np.random.normal(100, 20, num_patients),       # LDL
        np.random.normal(150, 30, num_patients),       # Triglycerides
        np.random.normal(13.5, 1.5, num_patients),     # Hemoglobin
        np.random.normal(40, 5, num_patients),         # Hematocrit
        np.random.normal(7000, 1500, num_patients),    # WBC Count
        np.random.normal(5, 0.5, num_patients),        # RBC Count
        np.random.normal(250000, 50000, num_patients), # Platelet Count
        np.random.normal(1, 0.2, num_patients),        # Creatinine
        np.random.normal(15, 5, num_patients),         # BUN
        np.random.normal(140, 5, num_patients),        # Sodium
        np.random.normal(4, 0.5, num_patients),        # Potassium
        np.random.normal(9.5, 0.5, num_patients),      # Calcium
        np.random.normal(2, 0.2, num_patients),        # Magnesium
        np.random.normal(50, 10, num_patients),        # Muscle Strength
        np.random.normal(30, 5, num_patients),         # Motor Function Score
        np.random.normal(60, 10, num_patients),        # Speech Clarity
        np.random.normal(40, 10, num_patients),        # Swallowing Function
        np.random.normal(30, 10, num_patients),        # Respiratory Capacity
    ])

    labels = np.concatenate([np.ones(num_patients // 2), np.zeros(num_patients - (num_patients // 2))])
    df = pd.DataFrame(data, columns=parameters)
    df['ALS'] = labels
    return df

# Load and preprocess the data
df = create_realistic_data()
X = df.drop(columns=['ALS'])
y = df['ALS']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=0),
    "Logistic Regression": LogisticRegression(random_state=0),
    "Support Vector Machine": SVC(probability=True, random_state=0),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME", random_state=0)
}

# Train models and calculate performance metrics
def train_and_evaluate_models(models, X_train, y_train, X_test, y_test):
    model_performance = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_prob),
            "MCC": matthews_corrcoef(y_test, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Cohen's Kappa": cohen_kappa_score(y_test, y_pred),
            "Brier Score": brier_score_loss(y_test, y_prob),
            "Logarithmic Loss": log_loss(y_test, y_prob),
            "F2 Score": fbeta_score(y_test, y_pred, beta=2),
            "Jaccard Index": jaccard_score(y_test, y_pred),
            "Hamming Loss": hamming_loss(y_test, y_pred),
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "ROC Curve": roc_curve(y_test, y_prob),
            "Precision-Recall Curve": precision_recall_curve(y_test, y_prob)
        }
        model_performance.append(metrics)
    return pd.DataFrame(model_performance)

performance_df = train_and_evaluate_models(models, X_train, y_train, X_test, y_test)

# Sidebar menu
st.sidebar.title("Menu Options")
menu_option = st.sidebar.radio("Choose an option", ["Welcome", "Data Input", "Model Information", "Graphs", "Accessibility Settings"])

# Welcome screen
if menu_option == "Welcome":
    st.write("# Welcome to Mitosense's ALS Detection Model!")
    st.markdown("""
    This application allows you to:
    - Upload patient data and predict the likelihood of ALS.
    - Compare various machine learning models and their performance.
    - Visualize performance metrics and feature importance.
    - Use the sidebar to navigate through the different sections.
    """)

# Save the Report to a PDF
def save_report_to_pdf(model_performance, performance_df, parameters, graph_options):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="ALS Detection Model Report", ln=True, align="C")

    pdf.cell(200, 10, txt="Model Performance Comparison", ln=True, align="L")
    performance_summary = performance_df.to_string(index=False)
    pdf.multi_cell(0, 10, performance_summary)

    temp_images = []

    for model_name, metrics in model_performance.items():
        if "Confusion Matrix" in graph_options:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(metrics["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix for {model_name}")
            temp_image_path = f"{model_name}_confusion_matrix.png"
            fig.savefig(temp_image_path, bbox_inches='tight')
            pdf.add_page()
            pdf.cell(200, 10, txt=f"Confusion Matrix for {model_name}", ln=True, align="L")
            pdf.image(temp_image_path, w=180)
            temp_images.append(temp_image_path)

        if "ROC Curve" in graph_options:
            fig, ax = plt.subplots(figsize=(8, 6))
            fpr, tpr, _ = metrics["ROC Curve"]
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['ROC AUC']:.2f})")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title(f"ROC Curve for {model_name}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            temp_image_path = f"{model_name}_roc_curve.png"
            fig.savefig(temp_image_path, bbox_inches='tight')
            pdf.add_page()
            pdf.cell(200, 10, txt=f"ROC Curve for {model_name}", ln=True, align="L")
            pdf.image(temp_image_path, w=180)
            temp_images.append(temp_image_path)

        if "Precision-Recall Curve" in graph_options:
            fig, ax = plt.subplots(figsize=(8, 6))
            precision, recall, _ = metrics["Precision-Recall Curve"]
            ax.plot(recall, precision, label=f"{model_name}")
            ax.set_title(f"Precision-Recall Curve for {model_name}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend(loc="lower left")
            temp_image_path = f"{model_name}_precision_recall_curve.png"
            fig.savefig(temp_image_path, bbox_inches='tight')
            pdf.add_page()
            pdf.cell(200, 10, txt=f"Precision-Recall Curve for {model_name}", ln=True, align="L")
            pdf.image(temp_image_path, w=180)
            temp_images.append(temp_image_path)

        if "Feature Importance" in graph_options and hasattr(metrics["model"], "feature_importances_"):
            feature_importance = pd.DataFrame({
                'Feature': parameters,
                'Importance': metrics["model"].feature_importances_
            }).sort_values(by='Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)
            ax.set_title(f"Feature Importance for {model_name}")
            temp_image_path = f"{model_name}_feature_importance.png"
            fig.savefig(temp_image_path, bbox_inches='tight')
            pdf.add_page()
            pdf.cell(200, 10, txt=f"Feature Importance for {model_name}", ln=True, align="L")
            pdf.image(temp_image_path, w=180)
            temp_images.append(temp_image_path)

    if "Model Performance Comparison" in graph_options:
        fig, ax = plt.subplots(figsize=(8, 6))
        performance_df.plot(kind="bar", x="Model", y=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"], ax=ax)
        ax.set_title("Model Performance Comparison")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=2)
        temp_image_path = "model_performance_comparison.png"
        fig.savefig(temp_image_path, bbox_inches='tight')
        pdf.add_page()
        pdf.cell(200, 10, txt="Model Performance Comparison", ln=True, align="L")
        pdf.image(temp_image_path, w=180)
        temp_images.append(temp_image_path)

        fig, ax = plt.subplots(figsize=(8, 6))
        performance_df.plot(kind="bar", x="Model", y=[
            "MCC", "Balanced Accuracy", "Cohen's Kappa", "Brier Score", "Logarithmic Loss", "F2 Score", 
            "Jaccard Index", "Hamming Loss"], ax=ax)
        ax.set_title("Additional Model Performance Comparison")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=2)
        temp_image_path = "additional_model_performance_comparison.png"
        fig.savefig(temp_image_path, bbox_inches='tight')
        pdf.add_page()
        pdf.cell(200, 10, txt="Additional Model Performance Comparison", ln=True, align="L")
        pdf.image(temp_image_path, w=180)
        temp_images.append(temp_image_path)

    pdf_output = BytesIO()
    pdf_output.write(pdf.output(dest='S').encode('latin1'))
    pdf_output.seek(0)

    st.sidebar.write("### Report saved successfully!")
    st.sidebar.download_button(label="Download the report", data=pdf_output, file_name="als_detection_model_report.pdf", mime="application/pdf")

    for temp_image_path in temp_images:
        os.remove(temp_image_path)

if st.sidebar.button("Save Report to PDF"):
    save_report_to_pdf(model_performance, performance_df, parameters, ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importance", "Model Performance Comparison"])

# Data Input
if menu_option == "Data Input":
    st.sidebar.header("Data Input Options")
    data_input_option = st.sidebar.radio("Select Data Input Method", ["Manual Entry", "CSV Upload", "Example Data"])

    if data_input_option == "Manual Entry":
        st.write("# Enter new patient data")
        uploaded_file = st.file_uploader("Upload CSV for one patient", type="csv")
        patient_df = None
        if uploaded_file is not None:
            patient_df = pd.read_csv(uploaded_file)
            if set(parameters).issubset(patient_df.columns):
                for param in parameters:
                    st.session_state[param] = float(patient_df[param].values[0])
            else:
                st.write("Error: The uploaded CSV file does not contain the required columns.")

        new_data = []
        for param in parameters:
            value = st.number_input(f"{param}", min_value=0.0, max_value=1000000000.0, value=st.session_state.get(param, 50.0))
            new_data.append(value)

        if uploaded_file is not None and patient_df is not None:
            st.write("### Uploaded Patient Data")
            st.dataframe(patient_df)

        if st.button("Predict ALS"):
            new_data = np.array(new_data).reshape(1, -1)
            new_data_scaled = scaler.transform(new_data)
            model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))
            model = models[model_choice]
            prediction = model.predict(new_data_scaled)[0]
            st.write("The patient is predicted to have ALS." if prediction == 1 else "The patient is predicted not to have ALS.")

    elif data_input_option == "CSV Upload":
        st.write("# Choose a CSV file")
        uploaded_file = st.file_uploader("Upload", type="csv")
        if uploaded_file is not None:
            new_data = pd.read_csv(uploaded_file)
            if set(parameters).issubset(new_data.columns):
                new_data_scaled = scaler.transform(new_data[parameters])
                model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))
                model = models[model_choice]
                predictions = model.predict(new_data_scaled)
                new_data['ALS Prediction'] = predictions
                st.write("Predictions for uploaded data:")
                st.dataframe(new_data)

                metrics_to_plot = st.multiselect("Select metrics to plot", ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"], default=[])
                additional_metrics_to_plot = st.multiselect("Select additional metrics to plot", [
                    "MCC", "Balanced Accuracy", "Cohen's Kappa", "Brier Score", "Logarithmic Loss", "F2 Score", "Jaccard Index", "Hamming Loss"], default=[])

                if st.button("Update Graphs"):
                    updated_performance_df = train_and_evaluate_models(models, X_train, y_train, new_data_scaled, predictions)
                    st.write("# Updated Model Performance Comparison")
                    st.dataframe(updated_performance_df)

                    best_model = updated_performance_df.loc[updated_performance_df["Accuracy"].idxmax()]
                    st.write(f"### Best Model: {best_model['Model']}")
                    st.write(f"Accuracy: {best_model['Accuracy']:.2f}")
                    st.write(f"Precision: {best_model['Precision']:.2f}")
                    st.write(f"Recall: {best_model['Recall']:.2f}")
                    st.write(f"F1 Score: {best_model['F1 Score']:.2f}")
                    st.write(f"ROC AUC: {best_model['ROC AUC']:.2f}")

                    if metrics_to_plot:
                        fig = px.bar(updated_performance_df, x="Model", y=metrics_to_plot, barmode="group")
                        st.plotly_chart(fig)

                    if additional_metrics_to_plot:
                        fig = px.bar(updated_performance_df, x="Model", y=additional_metrics_to_plot, barmode="group")
                        st.plotly_chart(fig)

    elif data_input_option == "Example Data":
        st.write("# View Example Patients")
        num_example_patients = st.number_input("Enter the number of example patients to view:", min_value=1, max_value=100, value=10, step=1)
        if st.button("Generate Example Data"):
            example_data = create_realistic_data(num_patients=int(num_example_patients))
            st.dataframe(example_data)
            example_data_scaled = scaler.transform(example_data[parameters])
            model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))
            model = models[model_choice]
            predictions = model.predict(example_data_scaled)
            example_data['ALS Prediction'] = predictions
            st.dataframe(example_data)

# Model Information
if menu_option == "Model Information":
    st.write("# Model Performance Comparison")
    st.dataframe(performance_df)

    best_model = performance_df.loc[performance_df["Accuracy"].idxmax()]
    st.write(f"### Best Model: {best_model['Model']}")
    st.write(f"Accuracy: {best_model['Accuracy']:.2f}")
    st.write(f"Precision: {best_model['Precision']:.2f}")
    st.write(f"Recall: {best_model['Recall']:.2f}")
    st.write(f"F1 Score: {best_model['F1 Score']:.2f}")
    st.write(f"ROC AUC: {best_model['ROC AUC']:.2f}")

    metrics_to_plot = st.multiselect("Select metrics to plot", ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"], default=[])
    if metrics_to_plot:
        fig = px.bar(performance_df, x="Model", y=metrics_to_plot, barmode="group")
        st.plotly_chart(fig)

    additional_metrics_to_plot = st.multiselect("Select additional metrics to plot", [
        "MCC", "Balanced Accuracy", "Cohen's Kappa", "Brier Score", "Logarithmic Loss", "F2 Score", "Jaccard Index", "Hamming Loss"], default=[])
    if additional_metrics_to_plot:
        fig = px.bar(performance_df, x="Model", y=additional_metrics_to_plot, barmode="group")
        st.plotly_chart(fig)

    if st.sidebar.checkbox("Show Metric Descriptions"):
        st.write("## Metric Descriptions")
        metric_descriptions = {
            "Accuracy": "Accuracy is the ratio of correctly predicted instances to the total instances.",
            "Precision": "Precision is the ratio of correctly predicted positive observations to the total predicted positives.",
            "Recall": "Recall is the ratio of correctly predicted positive observations to the all observations in actual class.",
            "F1 Score": "The F1 Score is the weighted average of Precision and Recall.",
            "ROC AUC": "ROC AUC is the Area Under the Receiver Operating Characteristic Curve. It represents the model's ability to distinguish between classes.",
            "MCC": "The Matthews Correlation Coefficient (MCC) is a measure of the quality of binary classifications.",
            "Balanced Accuracy": "Balanced Accuracy is the average of recall obtained on each class.",
            "Cohen's Kappa": "Cohen's Kappa measures the agreement between two raters who each classify N items into C mutually exclusive categories.",
            "Brier Score": "The Brier Score measures the mean squared difference between predicted probability and the actual outcome.",
            "Logarithmic Loss": "Logarithmic Loss, or Log Loss, measures the performance of a classification model where the prediction input is a probability value between 0 and 1.",
            "F2 Score": "The F2 Score is a weighted average of Precision and Recall, but with more weight given to Recall.",
            "Jaccard Index": "The Jaccard Index measures similarity between finite sample sets.",
            "Hamming Loss": "The Hamming Loss is the fraction of labels that are incorrectly predicted."
        }
        for metric, description in metric_descriptions.items():
            st.write(f"### {metric}")
            st.write(description)

# Graphs
if menu_option == "Graphs":
    st.write("# Graphs")
    st.sidebar.header("Graph Options")
    graph_options = st.sidebar.multiselect("Select Graphs", ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importance"], default=[])
    selected_model = st.sidebar.selectbox("Select Model for Graphs", list(models.keys()))
    show_all_models = st.sidebar.button("Show All Models for Selected Graphs")

    if st.sidebar.checkbox("Show Graph Descriptions"):
        st.write("## Graph Descriptions")
        graph_descriptions = {
            "Confusion Matrix": "A Confusion Matrix is a table used to evaluate the performance of a classification model.",
            "ROC Curve": "The ROC (Receiver Operating Characteristic) Curve is a graphical representation of a model's diagnostic ability.",
            "Precision-Recall Curve": "The Precision-Recall Curve is a plot that shows the trade-off between precision and recall for different threshold values.",
            "Feature Importance": "Feature Importance indicates the contribution of each feature to the prediction made by the model."
        }
        for graph, description in graph_descriptions.items():
            st.write(f"### {graph}")
            st.write(description)

    def plot_graphs(model_name, metrics):
        if "Confusion Matrix" in graph_options:
            st.write(f"### Confusion Matrix for {model_name}")
            fig, ax = plt.subplots()
            sns.heatmap(metrics["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix for {model_name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        if "ROC Curve" in graph_options:
            st.write(f"### ROC Curve for {model_name}")
            fig, ax = plt.subplots()
            fpr, tpr, _ = metrics["ROC Curve"]
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['ROC AUC']:.2f})")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title(f"ROC Curve for {model_name}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)

        if "Precision-Recall Curve" in graph_options:
            st.write(f"### Precision-Recall Curve for {model_name}")
            fig, ax = plt.subplots()
            precision, recall, _ = metrics["Precision-Recall Curve"]
            ax.plot(recall, precision, label=f"{model_name}")
            ax.set_title(f"Precision-Recall Curve for {model_name}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend(loc="lower left")
            st.pyplot(fig)

        if "Feature Importance" in graph_options and hasattr(metrics["model"], "feature_importances_"):
            st.write(f"### Feature Importance for {model_name}")
            feature_importance = pd.DataFrame({
                'Feature': parameters,
                'Importance': metrics["model"].feature_importances_
            }).sort_values(by='Importance', ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)
            ax.set_title(f"Feature Importance for {model_name}")
            st.pyplot(fig)

    if show_all_models:
        for model_name, metrics in performance_df.iterrows():
            plot_graphs(model_name, metrics)
    else:
        metrics = performance_df.loc[performance_df["Model"] == selected_model].squeeze()
        plot_graphs(selected_model, metrics)

# Accessibility Settings
if menu_option == "Accessibility Settings":
    st.sidebar.header("Accessibility Settings")
    font_size = st.sidebar.slider("Adjust Font Size", min_value=10, max_value=30, value=16)
    st.write(f"<style>body {{font-size: {font_size}px;}}</style>", unsafe_allow_html=True)

    color_theme = st.sidebar.radio("Select Color Theme", ["Default", "High Contrast", "Colorblind Friendly"])
    if color_theme == "High Contrast":
        st.write("<style>body {background-color: black; color: white;}</style>", unsafe_allow_html=True)
    elif color_theme == "Colorblind Friendly":
        st.write("<style>body {background-color: white; color: black;}}</style>", unsafe_allow_html=True)

    language = st.sidebar.radio("Select Language", ["English", "Spanish", "French"])
    if language == "Spanish":
        st.write("Idioma seleccionado: Español")
    elif language == "French":
        st.write("Langue sélectionnée: Français")
