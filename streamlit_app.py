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

# Load CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# Add logo
logo_path = "/OneDrive/mitosenseinc.com/Screenshots/Logo.PNG"
st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        justify-content: flex-start;
        align-items: center;
    }
    .logo-container img {
        width: 50px;  # Adjust the width as needed
        margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div class="logo-container">
        <img src="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,{st.file_uploader(logo_path)}" alt="Logo">
    </div>
    """,
    unsafe_allow_html=True
)

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

    # Ensure labels length matches num_patients
    half_patients = num_patients // 2
    labels = np.concatenate([np.ones(half_patients), np.zeros(num_patients - half_patients)])
    df = pd.DataFrame(data, columns=parameters)
    df['ALS'] = labels
    return df
    
# Load the data into a database
df = create_realistic_data()

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
menu_option = st.sidebar.radio("Choose an option", ["Data Input", "Model Information", "Graphs", "Accessibility Settings"])

if menu_option == "Data Input":
    st.sidebar.header("Data Input Options")
    data_input_option = st.sidebar.radio("Select Data Input Method", ["Manual Entry", "CSV Upload", "Example Data"])

    if data_input_option == "Manual Entry":
        st.write("# Enter new patient data")

        # File uploader for single patient CSV
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
    
        # Display the patient data below the number input options
        if uploaded_file is not None and patient_df is not None:
            st.write("### Uploaded Patient Data")
            st.dataframe(patient_df)
    
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
            else:
                st.write("Error: The uploaded CSV file does not contain the required columns.")

    elif data_input_option == "Example Data":
        st.write("# View Example Patients")
        
        num_example_patients = st.number_input("Enter the number of example patients to view:", min_value=1, max_value=100, value=10, step=1)
    
        if st.button("Generate Example Data"):
            # Ensure the number of patients is an integer
            num_example_patients = int(num_example_patients)
    
            example_data = create_realistic_data(num_patients=num_example_patients)
            st.dataframe(example_data)
            
            st.write("## Predictions for example data")
            example_data_scaled = scaler.transform(example_data[parameters])
            model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))
            model = models[model_choice]
            predictions = model.predict(example_data_scaled)
            example_data['ALS Prediction'] = predictions
            st.dataframe(example_data)


elif menu_option == "Model Information":
    st.write("# Model Performance Comparison")

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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=2)  # Adjust the legend position
    st.pyplot(fig)

elif menu_option == "Graphs":
    st.write("# Graphs")
    st.sidebar.header("Graph Options")
    graph_options = st.sidebar.multiselect("Select Graphs", ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importance"])
    show_graph_descriptions = st.sidebar.checkbox("Show Graph Descriptions")

    if show_graph_descriptions:
        st.write("## Graph Descriptions")
        
        st.write("### Confusion Matrix")
        st.write("""
        A Confusion Matrix is a table used to evaluate the performance of a classification model.
        It shows the actual versus predicted classifications and is useful for understanding the number of true positives, true negatives, false positives, and false negatives.
        """)

        st.write("### ROC Curve")
        st.write("""
        The ROC (Receiver Operating Characteristic) Curve is a graphical representation of a model's diagnostic ability.
        It plots the true positive rate (sensitivity) against the false positive rate (1 - specificity) at various threshold settings.
        The area under the curve (AUC) represents the model's ability to distinguish between classes.
        """)

        st.write("### Precision-Recall Curve")
        st.write("""
        The Precision-Recall Curve is a plot that shows the trade-off between precision and recall for different threshold values.
        Precision is the ratio of true positive predictions to the total predicted positives, while recall is the ratio of true positive predictions to all actual positives.
        This curve is particularly useful for imbalanced datasets.
        """)

        st.write("### Feature Importance")
        st.write("""
        Feature Importance indicates the contribution of each feature to the prediction made by the model.
        It helps in understanding which features are most influential in the model's decision-making process.
        This graph is typically available for tree-based models like Random Forest and Gradient Boosting.
        """)

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
    st.sidebar.header("Accessibility Settings")
    font_size = st.sidebar.slider("Adjust Font Size", min_value=10, max_value=30, value=16)
    st.write(f"<style>body {{font-size: {font_size}px;}}</style>", unsafe_allow_html=True)

    color_theme = st.sidebar.radio("Select Color Theme", ["Default", "High Contrast", "Colorblind Friendly"])
    if color_theme == "High Contrast":
        st.write("<style>body {background-color: black; color: white;}</style>", unsafe_allow_html=True)
    elif color_theme == "Colorblind Friendly":
        st.write("<style>body {background-color: white; color: black;}</style>", unsafe_allow_html=True)

    language = st.sidebar.radio("Select Language", ["English", "Spanish", "French"])
    if language == "Spanish":
        st.write("Idioma seleccionado: Español")
    elif language == "French":
        st.write("Langue sélectionnée: Français")
