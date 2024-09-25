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
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, roc_curve, precision_recall_curve, matthews_corrcoef, 
                             balanced_accuracy_score, cohen_kappa_score, brier_score_loss, 
                             log_loss, fbeta_score, jaccard_score, hamming_loss)
from fpdf import FPDF
from io import BytesIO
import base64
import os
import matplotlib.colors as mcolors

# Load CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

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

if 'sidebar_visible' not in st.session_state:
    st.session_state.sidebar_visible = False

def hide_sidebar():
    st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 0;
        margin-left: -300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        width: 0;
        margin-left: -300px;
    }
    </style>
    """, unsafe_allow_html=True)

def show_sidebar():
    st.markdown("""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 300px;
        margin-left: 0;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        width: 300px;
        margin-left: 0;
    }
    </style>
    """, unsafe_allow_html=True)

class ALSDetectionApp:
    def __init__(self):
        self.general_parameters = [
            'Heart Rate', 'Blood Pressure Systolic', 'Blood Pressure Diastolic',
            'Respiratory Rate', 'Oxygen Saturation', 'Temperature', 'Weight',
            'Height', 'BMI', 'Blood Glucose', 'Cholesterol', 'HDL', 'LDL',
            'Triglycerides', 'Hemoglobin', 'Hematocrit', 'WBC Count',
            'RBC Count', 'Platelet Count', 'Creatinine', 'BUN', 'Sodium',
            'Potassium', 'Calcium', 'Magnesium'
        ]

        self.als_specific_parameters = [
            'Muscle Strength', 'Motor Function Score', 'Speech Clarity',
            'Swallowing Function', 'Respiratory Capacity'
        ]

        self.parameters = self.general_parameters + self.als_specific_parameters
        self.df = pd.DataFrame()
        self.scaler = StandardScaler()
        self.performance_df = pd.DataFrame()

        self.models = {
            "Random Forest": RandomForestClassifier(random_state=0),
            "Logistic Regression": LogisticRegression(random_state=0),
            "Support Vector Machine": SVC(probability=True, random_state=0),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=0),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=0),
            "AdaBoost": AdaBoostClassifier(algorithm="SAMME", random_state=0)
        }
      
        self.model_performance = {}

    @st.cache_data
    def create_realistic_data(parameters, num_patients=1000):
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

    def load_data(self):
        self.df = ALSDetectionApp.create_realistic_data(self.parameters)

    def preprocess_data(self):
        X = self.df.drop(columns=['ALS'])
        y = self.df['ALS']
        X_scaled = self.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=0)

    def train_models(self, X_train, y_train, X_test, y_test):
        performance_metrics = []

        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)

            metrics = {
                "model": model,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "mcc": matthews_corrcoef(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "kappa": cohen_kappa_score(y_test, y_pred),
                "brier": brier_score_loss(y_test, y_prob),
                "logloss": log_loss(y_test, y_prob),
                "f2": fbeta_score(y_test, y_pred, beta=2),
                "jaccard": jaccard_score(y_test, y_pred),
                "hamming": hamming_loss(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "roc_curve": roc_curve(y_test, y_prob),
                "precision_recall_curve": precision_recall_curve(y_test, y_prob)
            }

            self.model_performance[model_name] = metrics

            performance_metrics.append({
                "Model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "mcc": metrics["mcc"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "kappa": metrics["kappa"],
                "brier": metrics["brier"],
                "logloss": metrics["logloss"],
                "f2": metrics["f2"],
                "jaccard": metrics["jaccard"],
                "hamming": metrics["hamming"]
            })

        self.performance_df = pd.DataFrame(performance_metrics)

   # def update_performance_df(self, new_data):
   #     self.performance_df = new_data

    def update_performance_df(self, new_data):
    # Ensure the new data contains all necessary parameters
      if set(self.parameters).issubset(new_data.columns):
          # Drop any additional columns such as 'ALS Prediction'
          if 'ALS Prediction' in new_data.columns:
              new_data = new_data.drop(columns=['ALS Prediction'])
  
          # Preprocess the new data and split it into training and testing sets
          X_new = new_data.drop(columns=['ALS'])
          y_new = new_data['ALS']
          X_new_scaled = self.scaler.transform(X_new)
          X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new_scaled, y_new, test_size=0.2, random_state=0)
          
          # Train the models on the new data
          self.train_models(X_train_new, y_train_new, X_test_new, y_test_new)
          
          # Update the performance DataFrame with the new data's model performance
          self.performance_df = pd.DataFrame([
              {
                  "Model": model_name,
                  "accuracy": metrics["accuracy"],
                  "precision": metrics["precision"],
                  "recall": metrics["recall"],
                  "f1": metrics["f1"],
                  "roc_auc": metrics["roc_auc"],
                  "mcc": metrics["mcc"],
                  "balanced_accuracy": metrics["balanced_accuracy"],
                  "kappa": metrics["kappa"],
                  "brier": metrics["brier"],
                  "logloss": metrics["logloss"],
                  "f2": metrics["f2"],
                  "jaccard": metrics["jaccard"],
                  "hamming": metrics["hamming"]
              } for model_name, metrics in self.model_performance.items()
          ])
      else:
          st.error("Uploaded data does not contain the necessary parameters.")

    def save_report_to_pdf(self):
        graph_options = ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importance"]
    
        pdf = FPDF()
        pdf.add_page()
    
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="ALS Detection Model Report - Graphs Only", ln=True, align="C")
    
        temp_images = []
    
        for model_name, metrics in self.model_performance.items():
            if "Confusion Matrix" in graph_options:
                fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
                sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"Confusion Matrix for {model_name}")
                temp_image_path = f"{model_name}_confusion_matrix.png"
                fig.savefig(temp_image_path, bbox_inches='tight')
                pdf.add_page()
                pdf.cell(200, 10, txt=f"Confusion Matrix for {model_name}", ln=True, align="L")
                pdf.image(temp_image_path, w=180)  # Adjust width as needed
                temp_images.append(temp_image_path)
    
            if "ROC Curve" in graph_options:
                fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
                fpr, tpr, _ = metrics["roc_curve"]
                ax.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['roc_auc']:.2f})")
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set_title(f"ROC Curve for {model_name}")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend(loc="lower right")
                temp_image_path = f"{model_name}_roc_curve.png"
                fig.savefig(temp_image_path, bbox_inches='tight')
                pdf.add_page()
                pdf.cell(200, 10, txt=f"ROC Curve for {model_name}", ln=True, align="L")
                pdf.image(temp_image_path, w=180)  # Adjust width as needed
                temp_images.append(temp_image_path)
    
            if "Precision-Recall Curve" in graph_options:
                fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
                precision, recall, _ = metrics["precision_recall_curve"]
                ax.plot(recall, precision, label=f"{model_name}")
                ax.set_title(f"Precision-Recall Curve for {model_name}")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.legend(loc="lower left")
                temp_image_path = f"{model_name}_precision_recall_curve.png"
                fig.savefig(temp_image_path, bbox_inches='tight')
                pdf.add_page()
                pdf.cell(200, 10, txt=f"Precision-Recall Curve for {model_name}", ln=True, align="L")
                pdf.image(temp_image_path, w=180)  # Adjust width as needed
                temp_images.append(temp_image_path)
    
            if "Feature Importance" in graph_options and hasattr(metrics["model"], "feature_importances_"):
                feature_importance = pd.DataFrame({
                    'Feature': self.parameters,
                    'Importance': metrics["model"].feature_importances_
                }).sort_values(by='Importance', ascending=False)
                fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
                sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)
                ax.set_title(f"Feature Importance for {model_name}")
                temp_image_path = f"{model_name}_feature_importance.png"
                fig.savefig(temp_image_path, bbox_inches='tight')
                pdf.add_page()
                pdf.cell(200, 10, txt=f"Feature Importance for {model_name}", ln=True, align="L")
                pdf.image(temp_image_path, w=180)  # Adjust width as needed
                temp_images.append(temp_image_path)
    
        # Include the bar graph for Model Performance Comparison with selected metrics (accuracy, precision, recall, f1, roc_auc)
        try:
            selected_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            fig, ax = plt.subplots(figsize=(8, 6))
            self.performance_df.set_index('Model')[selected_metrics].plot(kind='bar', ax=ax, color=list(mcolors.TABLEAU_COLORS.values()), edgecolor='black')
            ax.set_title("Model Performance Comparison (Selected Metrics)")
            ax.set_xlabel("Model")
            ax.set_ylabel("Scores")
            ax.legend(loc="best", bbox_to_anchor=(1, 1))
            temp_image_path = "model_performance_comparison_selected_metrics.png"
            fig.savefig(temp_image_path, bbox_inches='tight')
            pdf.add_page()
            pdf.cell(200, 10, txt="Model Performance Comparison (Selected Metrics)", ln=True, align="L")
            pdf.image(temp_image_path, w=180)  # Adjust width as needed
            temp_images.append(temp_image_path)
        except ValueError as e:
            st.error(f"Error generating bar graph image: {e}")
            st.write("Please ensure that the 'kaleido' package is installed by running `pip install -U kaleido`.")
    
        # Include the bar graphs for Model Performance Comparison excluding 'logloss'
        try:
            metrics_to_exclude = ['logloss']
            filtered_metrics = self.performance_df.drop(columns=metrics_to_exclude)
            fig, ax = plt.subplots(figsize=(8, 6))
            filtered_metrics.set_index('Model').plot(kind='bar', ax=ax, color=list(mcolors.TABLEAU_COLORS.values()), edgecolor='black')
            ax.set_title("Model Performance Comparison (Excluding Logloss)")
            ax.set_xlabel("Model")
            ax.set_ylabel("Scores")
            ax.legend(loc="best", bbox_to_anchor=(1, 1))
            temp_image_path = "model_performance_comparison_excluding_logloss.png"
            fig.savefig(temp_image_path, bbox_inches='tight')
            pdf.add_page()
            pdf.cell(200, 10, txt="Model Performance Comparison (Excluding Logloss)", ln=True, align="L")
            pdf.image(temp_image_path, w=180)  # Adjust width as needed
            temp_images.append(temp_image_path)
        except ValueError as e:
            st.error(f"Error generating bar graph image: {e}")
            st.write("Please ensure that the 'kaleido' package is installed by running `pip install -U kaleido`.")
    
        # Include the bar graph for all remaining metrics except accuracy, precision, recall, f1, roc_auc, and logloss
        try:
            remaining_metrics = ['mcc', 'balanced_accuracy', 'kappa', 'brier', 'f2', 'jaccard', 'hamming']
            filtered_remaining_metrics = self.performance_df[remaining_metrics]
            fig, ax = plt.subplots(figsize=(8, 6))
            filtered_remaining_metrics.set_index('Model').plot(kind='bar', ax=ax, color=list(mcolors.TABLEAU_COLORS.values()), edgecolor='black')
            ax.set_title("Model Performance Comparison (Remaining Metrics)")
            ax.set_xlabel("Model")
            ax.set_ylabel("Scores")
            ax.legend(loc="best", bbox_to_anchor=(1, 1))
            temp_image_path = "model_performance_comparison_remaining_metrics.png"
            fig.savefig(temp_image_path, bbox_inches='tight')
            pdf.add_page()
            pdf.cell(200, 10, txt="Model Performance Comparison (Remaining Metrics)", ln=True, align="L")
            pdf.image(temp_image_path, w=180)  # Adjust width as needed
            temp_images.append(temp_image_path)
        except ValueError as e:
            st.error(f"Error generating bar graph image: {e}")
            st.write("Please ensure that the 'kaleido' package is installed by running `pip install -U kaleido`.")
        
        # Include the bar graph for the logloss metric only
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            self.performance_df[['logloss']].plot(kind='bar', ax=ax, color=mcolors.TABLEAU_COLORS.values(), edgecolor='black')
            ax.set_title("Model Performance Comparison (Logloss Only)")
            ax.set_xlabel("Model")
            ax.set_ylabel("Logloss")
            ax.legend(loc="best", bbox_to_anchor=(1, 1))
            temp_image_path = "model_performance_comparison_logloss.png"
            fig.savefig(temp_image_path, bbox_inches='tight')
            pdf.add_page()
            pdf.cell(200, 10, txt="Model Performance Comparison (Logloss Only)", ln=True, align="L")
            pdf.image(temp_image_path, w=180)  # Adjust width as needed
            temp_images.append(temp_image_path)
        except ValueError as e:
            st.error(f"Error generating bar graph image: {e}")
            st.write("Please ensure that the 'kaleido' package is installed by running `pip install -U kaleido`.")
    
        pdf_output = BytesIO()
        pdf_output.write(pdf.output(dest='S').encode('latin1'))
        pdf_output.seek(0)
      
        st.sidebar.write("### Report saved successfully!")
        st.sidebar.download_button(label="Download the report", data=pdf_output, file_name="als_detection_model_graphs_report.pdf", mime="application/pdf")
      
        for temp_image_path in temp_images:
            os.remove(temp_image_path)

    def run(self):
        if st.session_state.sidebar_visible:
            show_sidebar()
        else:
            hide_sidebar()

        st.sidebar.title("Menu Options")
        menu_option = st.sidebar.radio("Choose an option", ["Welcome", "Data Input", "Model Information", "Graphs", "Accessibility Settings"])

        if menu_option == "Welcome":
            self.display_welcome()
        elif menu_option == "Data Input":
            self.display_data_input()
        elif menu_option == "Model Information":
            self.display_model_information()
        elif menu_option == "Graphs":
            self.display_graphs()
        elif menu_option == "Accessibility Settings":
            self.display_accessibility_settings()

        if st.sidebar.button("Save Report to PDF"):
            self.save_report_to_pdf()

    def display_welcome(self):
        st.write("# Welcome to Mitosense's ALS Detection Model!")
        st.markdown("""
        This application allows you to:
        - Upload patient data and predict the likelihood of ALS.
        - Compare various machine learning models and their performance.
        - Visualize performance metrics and feature importance.
        - Use the sidebar to navigate through the different sections.
        """)

        if st.button("Get Started"):
          st.session_state.sidebar_visible = True

    def display_data_input(self):
        st.sidebar.header("Data Input Options")
        data_input_option = st.sidebar.radio("Select Data Input Method", ["Manual Entry", "File Upload", "Example Data"])

        if data_input_option == "Manual Entry":
            self.display_manual_entry()
        elif data_input_option == "File Upload":
            self.display_file_to_upload()
        elif data_input_option == "Example Data":
            self.display_example_data()

    def display_manual_entry(self):
        st.write("# Enter new patient data")
        new_data = []
        for param in self.parameters:
            value = st.number_input(f"{param}", min_value=0.0, max_value=1000000000.0, value=50.0)
            new_data.append(value)

        if st.button("Predict ALS"):
            new_data = np.array(new_data).reshape(1, -1)
            new_data_scaled = self.scaler.transform(new_data)
            model_choice = st.sidebar.selectbox("Choose a model", list(self.models.keys()))
            model = self.models[model_choice]
            prediction = model.predict(new_data_scaled)[0]
            if prediction == 1:
                st.write("The patient is predicted to have ALS.")
            else:
                st.write("The patient is predicted not to have ALS.")

    def display_file_to_upload(self):
        st.write("# Choose a file (CSV or Excel)")
        uploaded_file = st.file_uploader("Upload", type=["csv", "xlsx"])
        if uploaded_file is not None:
            # Check file type
            if uploaded_file.name.endswith(".csv"):
                new_data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                new_data = pd.read_excel(uploaded_file)
    
            if set(self.parameters).issubset(new_data.columns):
                self.update_performance_df(new_data)
                new_data_scaled = self.scaler.transform(new_data[self.parameters])
                model_choice = st.sidebar.selectbox("Choose a model", list(self.models.keys()))
                model = self.models[model_choice]
                predictions = model.predict(new_data_scaled)
                new_data['ALS Prediction'] = predictions
                st.write("Predictions for uploaded data:")
                st.dataframe(new_data)
                # self.update_performance_df(new_data)
            else:
                st.error("Uploaded file does not contain the necessary parameters.")

    def display_example_data(self):
        st.write("# View Example Patients")
        num_example_patients = st.number_input("Enter the number of example patients to view:", min_value=1, max_value=100, value=10, step=1)
        if st.button("Generate Example Data"):
            example_data = ALSDetectionApp.create_realistic_data(self.parameters, num_patients=int(num_example_patients))
            example_data_scaled = self.scaler.transform(example_data[self.parameters])
            model_choice = st.sidebar.selectbox("Choose a model", list(self.models.keys()))
            model = self.models[model_choice]
            predictions = model.predict(example_data_scaled)
            example_data['ALS Prediction'] = predictions
            st.dataframe(example_data)
  
    def display_model_information(self):
        st.write("# Model Performance Comparison")
        st.dataframe(self.performance_df)
    
        best_model = self.performance_df.loc[self.performance_df["accuracy"].idxmax()]
        st.write(f"### Best Model: {best_model['Model']}")
        st.write(f"Accuracy: {best_model['accuracy']:.2f}")
        st.write(f"Precision: {best_model['precision']:.2f}")
        st.write(f"Recall: {best_model['recall']:.2f}")
        st.write(f"F1 Score: {best_model['f1']:.2f}")
        st.write(f"ROC AUC: {best_model['roc_auc']:.2f}")

        st.write("### Plotting the Model Performance Comparison")
        metrics_to_plot = st.multiselect("Select metrics to plot", ["accuracy", "precision", "recall", "f1", "roc_auc"], default=[])
        if metrics_to_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.performance_df.set_index('Model')[metrics_to_plot].plot(kind='bar', ax=ax, color=list(mcolors.TABLEAU_COLORS.values()), edgecolor='black')
            ax.set_title("Model Performance Comparison")
            ax.set_xlabel("Model")
            ax.set_ylabel("Scores")
            ax.legend(loc="best", bbox_to_anchor=(1, 1))
            st.pyplot(fig)

        st.write("### Additional Model Performance Comparison")
        additional_metrics_to_plot = st.multiselect("Select additional metrics to plot", ["mcc", "balanced_accuracy", "kappa", "brier", "logloss", "f2", "jaccard", "hamming"], default=[])
        if additional_metrics_to_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            self.performance_df.set_index('Model')[additional_metrics_to_plot].plot(kind='bar', ax=ax, color=list(mcolors.TABLEAU_COLORS.values()), edgecolor='black')
            ax.set_title("Additional Model Performance Comparison")
            ax.set_xlabel("Model")
            ax.set_ylabel("Scores")
            ax.legend(loc="best", bbox_to_anchor=(1, 1))
            st.pyplot(fig)

        show_metric_descriptions = st.sidebar.checkbox("Show Metric Descriptions")
        if show_metric_descriptions:
            self.display_metric_descriptions()

    def display_graphs(self):
        st.write("# Graphs")
        st.sidebar.header("Graph Options")
        graph_options = st.sidebar.multiselect("Select Graphs", ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importance"], default=[])
        selected_model = st.sidebar.selectbox("Select Model for Graphs", list(self.models.keys()))
        show_all_models = st.sidebar.button("Show All Models for Selected Graphs")
        show_graph_descriptions = st.sidebar.checkbox("Show Graph Descriptions")

        if show_graph_descriptions:
            self.display_graph_descriptions()

        if show_all_models:
            for model_name, metrics in self.model_performance.items():
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
                        'Feature': self.parameters,
                        'Importance': metrics["model"].feature_importances_
                    }).sort_values(by='Importance', ascending=False)
                    fig, ax = plt.subplots()
                    sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)
                    ax.set_title(f"Feature Importance for {model_name}")
                    st.pyplot(fig)
        else:
            metrics = self.model_performance[selected_model]
            if "Confusion Matrix" in graph_options:
                st.write(f"### Confusion Matrix for {selected_model}")
                fig, ax = plt.subplots()
                sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"Confusion Matrix for {selected_model}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            if "ROC Curve" in graph_options:
                st.write(f"### ROC Curve for {selected_model}")
                fig, ax = plt.subplots()
                fpr, tpr, _ = metrics["roc_curve"]
                ax.plot(fpr, tpr, label=f"{selected_model} (AUC = {metrics['roc_auc']:.2f})")
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set_title(f"ROC Curve for {selected_model}")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.legend(loc="lower right")
                st.pyplot(fig)

            if "Precision-Recall Curve" in graph_options:
                st.write(f"### Precision-Recall Curve for {selected_model}")
                fig, ax = plt.subplots()
                precision, recall, _ = metrics["precision_recall_curve"]
                ax.plot(recall, precision, label=f"{selected_model}")
                ax.set_title(f"Precision-Recall Curve for {selected_model}")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.legend(loc="lower left")
                st.pyplot(fig)

            if "Feature Importance" in graph_options and hasattr(metrics["model"], "feature_importances_"):
                st.write(f"### Feature Importance for {selected_model}")
                feature_importance = pd.DataFrame({
                    'Feature': self.parameters,
                    'Importance': metrics["model"].feature_importances_
                }).sort_values(by='Importance', ascending=False)
                fig, ax = plt.subplots()
                sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)
                ax.set_title(f"Feature Importance for {selected_model}")
                st.pyplot(fig)

    def display_accessibility_settings(self):
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

    def display_metric_descriptions(self):
        st.write("## Metric Descriptions")
        st.write("### Accuracy")
        st.write("""
        Accuracy is the ratio of correctly predicted instances to the total instances.
        """)
        st.write("### Precision")
        st.write("""
        Precision is the ratio of correctly predicted positive observations to the total predicted positives.
        """)
        st.write("### Recall")
        st.write("""
        Recall is the ratio of correctly predicted positive observations to the all observations in actual class.
        """)
        st.write("### F1 Score")
        st.write("""
        The F1 Score is the weighted average of Precision and Recall.
        """)
        st.write("### ROC AUC")
        st.write("""
        ROC AUC is the Area Under the Receiver Operating Characteristic Curve. It represents the model's ability to distinguish between classes.
        """)
        st.write("### MCC")
        st.write("""
        The Matthews Correlation Coefficient (MCC) is a measure of the quality of binary classifications.
        """)
        st.write("### Balanced Accuracy")
        st.write("""
        Balanced Accuracy is the average of recall obtained on each class.
        """)
        st.write("### Cohen's Kappa")
        st.write("""
        Cohen's Kappa measures the agreement between two raters who each classify N items into C mutually exclusive categories.
        """)
        st.write("### Brier Score")
        st.write("""
        The Brier Score measures the mean squared difference between predicted probability and the actual outcome.
        """)
        st.write("### Logarithmic Loss")
        st.write("""
        Logarithmic Loss, or Log Loss, measures the performance of a classification model where the prediction input is a probability value between 0 and 1.
        """)
        st.write("### F2 Score")
        st.write("""
        The F2 Score is a weighted average of Precision and Recall, but with more weight given to Recall.
        """)
        st.write("### Jaccard Index")
        st.write("""
        The Jaccard Index measures similarity between finite sample sets.
        """)
        st.write("### Hamming Loss")
        st.write("""
        The Hamming Loss is the fraction of labels that are incorrectly predicted.
        """)

    def display_graph_descriptions(self):
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

if __name__ == "__main__":
    app = ALSDetectionApp()
    app.load_data()
    X_train, X_test, y_train, y_test = app.preprocess_data()
    app.train_models(X_train, y_train, X_test, y_test)
    app.run()
