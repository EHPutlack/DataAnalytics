import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from fpdf import FPDF
from io import BytesIO
import os

# Define global list of parameters
general_parameters = [
    'Heart Rate', 'Blood Pressure Systolic', 'Blood Pressure Diastolic',
    'Respiratory Rate', 'Oxygen Saturation', 'Temperature', 'Weight',
    'Height', 'BMI', 'Blood Glucose', 'Cholesterol', 'HDL', 'LDL',
    'Triglycerides', 'Hemoglobin', 'Hematocrit', 'WBC Count',
    'RBC Count', 'Platelet Count', 'Creatinine', 'BUN', 'Sodium',
    'Potassium', 'Calcium', 'Magnesium'
]

als_specific_parameters = [
    'Muscle Strength', 'Motor Function Score', 'Speech Clarity',
    'Swallowing Function', 'Respiratory Capacity'
]

parameters = general_parameters + als_specific_parameters

# Function to create realistic fake data with ALS-specific parameters
@st.cache_data
def create_realistic_data(num_patients=1000):
    np.random.seed(0)

    # Generating realistic data distributions
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

    labels = np.concatenate([np.ones(num_patients // 2), np.zeros(num_patients // 2)])
    df = pd.DataFrame(data, columns=parameters)
    df['ALS'] = labels
    return df

# Load and display the realistic fake data
df = create_realistic_data()
st.dataframe(df.head())

# Split the data into training and test sets
X = df.drop(columns=['ALS'])
y = df['ALS']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Define the models to compare
models = {
    "Random Forest": RandomForestClassifier(random_state=0),
    "Logistic Regression": LogisticRegression(random_state=0),
    "Support Vector Machine": SVC(probability=True, random_state=0),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME", random_state=0)  # Updated to use SAMME algorithm
}

# Train models and calculate performance metrics
model_performance = {}
performance_metrics = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    model_performance[model_name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": roc_curve(y_test, y_prob),
        "precision_recall_curve": precision_recall_curve(y_test, y_prob)
    }

    performance_metrics.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

performance_df = pd.DataFrame(performance_metrics)  # Ensure this is defined before using it

# Sidebar menu
st.sidebar.title("Menu Options")
menu_option = st.sidebar.selectbox("Choose an option", ["Data Input Options", "Model Information", "Graphs", "Accessibility Settings"])

if menu_option == "Data Input Options":
    data_input_option = st.sidebar.selectbox("Select Data Input Method", ["Manual Entry", "CSV Upload", "Example Data"])

    if data_input_option == "Manual Entry":
        st.write("## Enter new patient data")

        # File uploader for single patient CSV
        uploaded_file = st.file_uploader("Upload CSV for one patient", type="csv")
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

        if st.button("Predict ALS"):
            new_data = np.array(new_data).reshape(1, -1)
            new_data_scaled = scaler.transform(new_data)
            model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))
            model = models[model_choice]
            prediction = model.predict(new_data_scaled)[0]
            if prediction == 1:
                st.write("The patient is predicted to have ALS.")
            else:
                st.write("The patient is predicted not to have ALS.")

    elif data_input_option == "CSV Upload":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

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
            else:
                st.write("Error: The uploaded CSV file does not contain the required columns.")

    elif data_input_option == "Example Data":
        st.write("Loading example data...")
        example_data = create_realistic_data(10)
        st.dataframe(example_data)
        st.write("## Predictions for example data")
        example_data_scaled = scaler.transform(example_data[parameters])
        model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))
        model = models[model_choice]
        predictions = model.predict(example_data_scaled)
        example_data['ALS Prediction'] = predictions
        st.dataframe(example_data)

elif menu_option == "Model Information":
    st.write("## Model Performance Comparison")

    st.dataframe(performance_df)

    best_model = performance_df.loc[performance_df["Accuracy"].idxmax()]
    st.write(f"### Best Model: {best_model['Model']}")
    st.write(f"Accuracy: {best_model['Accuracy']:.2f}")
    st.write(f"Precision: {best_model['Precision']:.2f}")
    st.write(f"Recall: {best_model['Recall']:.2f}")
    st.write(f"F1 Score: {best_model['F1 Score']:.2f}")
    st.write(f"ROC AUC: {best_model['ROC AUC']:.2f}")

    st.write("### Plotting the Model Performance Comparison")
    fig, ax = plt.subplots()
    performance_df.plot(kind="bar", x="Model", y=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"], ax=ax)
    ax.set_title("Model Performance Comparison")
    st.pyplot(fig)

elif menu_option == "Graphs":
    st.write("## Select Graphs to Display")
    graph_options = st.sidebar.multiselect("Select Graphs", ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importance"])

    for model_name, metrics in model_performance.items():
        if "Confusion Matrix" in graph_options:
            st.write(f"### Confusion Matrix for {model_name}")
            fig, ax = plt.subplots()
            sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix for {model_name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        if "ROC Curve" in graph_options:
            st.write(f"### ROC Curve for {model_name}")
            fig, ax = plt.subplots()
            fpr, tpr, _ = metrics["roc_curve"]
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['roc_auc']:.2f})")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title(f"ROC Curve for {model_name}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            st.pyplot(fig)

        if "Precision-Recall Curve" in graph_options:
            st.write(f"### Precision-Recall Curve for {model_name}")
            fig, ax = plt.subplots()
            precision, recall, _ = metrics["precision_recall_curve"]
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

# Allow user to save a comprehensive report to PDF
if st.sidebar.button("Save Report to PDF"):
    graph_options = st.sidebar.multiselect("Select Graphs for PDF", ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importance", "Model Performance Comparison"])

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'ALS Detection Model Report', 0, 1, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, 0, 1, 'L')
            self.ln(10)

        def chapter_body(self, body):
            self.set_font('Arial', '', 12)
            self.multi_cell(0, 10, body)
            self.ln()

        def add_image(self, image_path):
            self.image(image_path, x=None, y=None, w=150, h=150)

    pdf = PDF()
    pdf.add_page()

    pdf.chapter_title("Model Performance Comparison")
    performance_summary = pd.DataFrame(performance_metrics).to_string(index=False)
    pdf.chapter_body(performance_summary)

    temp_images = []

    for model_name, metrics in model_performance.items():
        if "Confusion Matrix" in graph_options:
            fig, ax = plt.subplots()
            sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix for {model_name}")
            temp_image_path = f"{model_name}_confusion_matrix.png"
            fig.savefig(temp_image_path)
            pdf.add_page()
            pdf.chapter_title(f"Confusion Matrix for {model_name}")
            pdf.add_image(temp_image_path)
            temp_images.append(temp_image_path)

        if "ROC Curve" in graph_options:
            fig, ax = plt.subplots()
            fpr, tpr, _ = metrics["roc_curve"]
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['roc_auc']:.2f})")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title(f"ROC Curve for {model_name}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            temp_image_path = f"{model_name}_roc_curve.png"
            fig.savefig(temp_image_path)
            pdf.add_page()
            pdf.chapter_title(f"ROC Curve for {model_name}")
            pdf.add_image(temp_image_path)
            temp_images.append(temp_image_path)

        if "Precision-Recall Curve" in graph_options:
            fig, ax = plt.subplots()
            precision, recall, _ = metrics["precision_recall_curve"]
            ax.plot(recall, precision, label=f"{model_name}")
            ax.set_title(f"Precision-Recall Curve for {model_name}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend(loc="lower left")
            temp_image_path = f"{model_name}_precision_recall_curve.png"
            fig.savefig(temp_image_path)
            pdf.add_page()
            pdf.chapter_title(f"Precision-Recall Curve for {model_name}")
            pdf.add_image(temp_image_path)
            temp_images.append(temp_image_path)

        if "Feature Importance" in graph_options and hasattr(metrics["model"], "feature_importances_"):
            feature_importance = pd.DataFrame({
                'Feature': parameters,
                'Importance': metrics["model"].feature_importances_
            }).sort_values(by='Importance', ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)
            ax.set_title(f"Feature Importance for {model_name}")
            temp_image_path = f"{model_name}_feature_importance.png"
            fig.savefig(temp_image_path)
            pdf.add_page()
            pdf.chapter_title(f"Feature Importance for {model_name}")
            pdf.add_image(temp_image_path)
            temp_images.append(temp_image_path)

    if "Model Performance Comparison" in graph_options:
        fig, ax = plt.subplots()
        performance_df.plot(kind="bar", x="Model", y=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"], ax=ax)
        ax.set_title("Model Performance Comparison")
        temp_image_path = "model_performance_comparison.png"
        fig.savefig(temp_image_path)
        pdf.add_page()
        pdf.chapter_title("Model Performance Comparison")
        pdf.add_image(temp_image_path)
        temp_images.append(temp_image_path)

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    st.write("### Report saved successfully!")
    st.download_button(label="Download the report", data=pdf_output, file_name="als_detection_model_report.pdf", mime="application/pdf")

    for temp_image_path in temp_images:
        os.remove(temp_image_path)

elif menu_option == "Accessibility Settings":
    font_size = st.sidebar.slider("Adjust Font Size", min_value=10, max_value=30, value=16)
    st.write(f"<style>body {{font-size: {font_size}px;}}</style>", unsafe_allow_html=True)

    color_theme = st.sidebar.selectbox("Select Color Theme", ["Default", "High Contrast", "Colorblind Friendly"])
    if color_theme == "High Contrast":
        st.write("<style>body {background-color: black; color: white;}</style>", unsafe_allow_html=True)
    elif color_theme == "Colorblind Friendly":
        st.write("<style>body {background-color: white; color: black;}</style>", unsafe_allow_html=True)

    language = st.sidebar.selectbox("Select Language", ["English", "Spanish", "French"])
    if language == "Spanish":
        st.write("Idioma seleccionado: Español")
    elif language == "French":
        st.write("Langue sélectionnée: Français")
