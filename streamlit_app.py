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

class ALSDetectionApp:
    def __init__(self):
        self.general_parameters = self.define_general_parameters()
        self.als_specific_parameters = self.define_als_specific_parameters()
        self.parameters = self.general_parameters + self.als_specific_parameters
        self.models = self.define_models()
        self.scaler = StandardScaler()
        self.performance_df = pd.DataFrame()
        self.model_performance = {}
        self.load_css("styles.css")
        self.add_logo("Logo.PNG")

    @staticmethod
    def define_general_parameters():
        return [
            'Heart Rate', 'Blood Pressure Systolic', 'Blood Pressure Diastolic',
            'Respiratory Rate', 'Oxygen Saturation', 'Temperature', 'Weight',
            'Height', 'BMI', 'Blood Glucose', 'Cholesterol', 'HDL', 'LDL',
            'Triglycerides', 'Hemoglobin', 'Hematocrit', 'WBC Count',
            'RBC Count', 'Platelet Count', 'Creatinine', 'BUN', 'Sodium',
            'Potassium', 'Calcium', 'Magnesium'
        ]

    @staticmethod
    def define_als_specific_parameters():
        return [
            'Muscle Strength', 'Motor Function Score', 'Speech Clarity',
            'Swallowing Function', 'Respiratory Capacity'
        ]

    @staticmethod
    def define_models():
        return {
            "Random Forest": RandomForestClassifier(random_state=0),
            "Logistic Regression": LogisticRegression(random_state=0),
            "Support Vector Machine": SVC(probability=True, random_state=0),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=0),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=0),
            "AdaBoost": AdaBoostClassifier(algorithm="SAMME", random_state=0)
        }

    @staticmethod
    def load_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    @staticmethod
    def img_to_base64(file_path):
        with open(file_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    def add_logo(self, logo_file_path):
        logo_base64 = self.img_to_base64(logo_file_path)
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

    @st.cache_data
    def create_realistic_data(_self, num_patients=1000):
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

        half_patients = num_patients // 2
        labels = np.concatenate([np.ones(half_patients), np.zeros(num_patients - half_patients)])
        df = pd.DataFrame(data, columns=_self.parameters)
        df['ALS'] = labels
        return df

    def load_data(self):
        self.df = self.create_realistic_data()

    def preprocess_data(self):
        X = self.df.drop(columns=['ALS'])
        y = self.df['ALS']
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train, X_test, y_test):
        performance_metrics = []

        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)

            metrics = self.calculate_metrics(y_test, y_pred, y_prob)
            metrics['model'] = model
            self.model_performance[model_name] = metrics

            performance_metrics.append({
                "Model": model_name,
                **{k: v for k, v in metrics.items() if k not in ['model', 'confusion_matrix', 'roc_curve', 'precision_recall_curve']}
            })

        self.performance_df = pd.DataFrame(performance_metrics)
        
        # Debugging step: print the columns of the DataFrame
        st.write("Columns in performance_df:", self.performance_df.columns.tolist())

    @staticmethod
    def calculate_metrics(y_test, y_pred, y_prob):
        return {
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

    def run(self):
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

    def save_report_to_pdf(self):
        graph_options = ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importance", "Model Performance Comparison"]

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="ALS Detection Model Report", ln=True, align="C")

        pdf.cell(200, 10, txt="Model Performance Comparison", ln=True, align="L")
        performance_summary = self.performance_df.to_string(index=False)
        pdf.multi_cell(0, 10, performance_summary)

        temp_images = []

        for model_name, metrics in self.model_performance.items():
            if "Confusion Matrix" in graph_options:
                temp_image_path = self.save_confusion_matrix(metrics, model_name)
                self.add_image_to_pdf(pdf, temp_image_path, f"Confusion Matrix for {model_name}")
                temp_images.append(temp_image_path)

            if "ROC Curve" in graph_options:
                temp_image_path = self.save_roc_curve(metrics, model_name)
                self.add_image_to_pdf(pdf, temp_image_path, f"ROC Curve for {model_name}")
                temp_images.append(temp_image_path)

            if "Precision-Recall Curve" in graph_options:
                temp_image_path = self.save_precision_recall_curve(metrics, model_name)
                self.add_image_to_pdf(pdf, temp_image_path, f"Precision-Recall Curve for {model_name}")
                temp_images.append(temp_image_path)

            if "Feature Importance" in graph_options and hasattr(metrics["model"], "feature_importances_"):
                temp_image_path = self.save_feature_importance(metrics, model_name)
                self.add_image_to_pdf(pdf, temp_image_path, f"Feature Importance for {model_name}")
                temp_images.append(temp_image_path)

        self.add_performance_comparison_to_pdf(pdf, graph_options, temp_images)

        pdf_output = BytesIO()
        pdf_output.write(pdf.output(dest='S').encode('latin1'))
        pdf_output.seek(0)

        st.sidebar.write("### Report saved successfully!")
        st.sidebar.download_button(label="Download the report", data=pdf_output, file_name="als_detection_model_report.pdf", mime="application/pdf")

        self.cleanup_temp_images(temp_images)

    def save_confusion_matrix(self, metrics, model_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix for {model_name}")
        temp_image_path = f"{model_name}_confusion_matrix.png"
        fig.savefig(temp_image_path, bbox_inches='tight')
        return temp_image_path

    def save_roc_curve(self, metrics, model_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = metrics["roc_curve"]
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['roc_auc']:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(f"ROC Curve for {model_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        temp_image_path = f"{model_name}_roc_curve.png"
        fig.savefig(temp_image_path, bbox_inches='tight')
        return temp_image_path

    def save_precision_recall_curve(self, metrics, model_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        precision, recall, _ = metrics["precision_recall_curve"]
        ax.plot(recall, precision, label=f"{model_name}")
        ax.set_title(f"Precision-Recall Curve for {model_name}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left")
        temp_image_path = f"{model_name}_precision_recall_curve.png"
        fig.savefig(temp_image_path, bbox_inches='tight')
        return temp_image_path

    def save_feature_importance(self, metrics, model_name):
        feature_importance = pd.DataFrame({
            'Feature': self.parameters,
            'Importance': metrics["model"].feature_importances_
        }).sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=ax)
        ax.set_title(f"Feature Importance for {model_name}")
        temp_image_path = f"{model_name}_feature_importance.png"
        fig.savefig(temp_image_path, bbox_inches='tight')
        return temp_image_path

    def add_image_to_pdf(self, pdf, image_path, title):
        pdf.add_page()
        pdf.cell(200, 10, txt=title, ln=True, align="L")
        pdf.image(image_path, w=180)

    def add_performance_comparison_to_pdf(self, pdf, graph_options, temp_images):
        if "Model Performance Comparison" in graph_options:
            self.plot_and_save_performance_comparison(pdf, "Model Performance Comparison", ["accuracy", "precision", "recall", "f1", "roc_auc"], "model_performance_comparison.png", temp_images)
            self.plot_and_save_performance_comparison(pdf, "Additional Model Performance Comparison", ["mcc", "balanced_accuracy", "kappa", "brier", "logloss", "f2", "jaccard", "hamming"], "additional_model_performance_comparison.png", temp_images)

    def plot_and_save_performance_comparison(self, pdf, title, metrics, image_name, temp_images):
        fig, ax = plt.subplots(figsize=(8, 6))
        self.performance_df.plot(kind="bar", x="Model", y=metrics, ax=ax)
        ax.set_title(title)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=2)
        temp_image_path = image_name
        fig.savefig(temp_image_path, bbox_inches='tight')
        self.add_image_to_pdf(pdf, temp_image_path, title)
        temp_images.append(temp_image_path)

    @staticmethod
    def cleanup_temp_images(temp_images):
        for temp_image_path in temp_images:
            os.remove(temp_image_path)

    def display_welcome(self):
        st.write("# Welcome to Mitosense's ALS Detection Model!")
        st.markdown("""
        This application allows you to:
        - Upload patient data and predict the likelihood of ALS.
        - Compare various machine learning models and their performance.
        - Visualize performance metrics and feature importance.
        - Use the sidebar to navigate through the different sections.
        """)

    def display_data_input(self):
        st.sidebar.header("Data Input Options")
        data_input_option = st.sidebar.radio("Select Data Input Method", ["Manual Entry", "CSV Upload", "Example Data"])

        if data_input_option == "Manual Entry":
            self.display_manual_entry()
        elif data_input_option == "CSV Upload":
            self.display_csv_upload()
        elif data_input_option == "Example Data":
            self.display_example_data()

    def display_manual_entry(self):
        st.write("# Enter new patient data")
        uploaded_file = st.file_uploader("Upload CSV for one patient", type="csv")

        if uploaded_file is not None:
            patient_df = pd.read_csv(uploaded_file)
            if set(self.parameters).issubset(patient_df.columns):
                for param in self.parameters:
                    st.session_state[param] = float(patient_df[param].values[0])
            else:
                st.write("Error: The uploaded CSV file does not contain the required columns.")
        
        new_data = [st.number_input(f"{param}", min_value=0.0, max_value=1000000000.0, value=st.session_state.get(param, 50.0)) for param in self.parameters]

        if uploaded_file is not None and patient_df is not None:
            st.write("### Uploaded Patient Data")
            st.dataframe(patient_df)

        if st.button("Predict ALS"):
            new_data = np.array(new_data).reshape(1, -1)
            new_data_scaled = self.scaler.transform(new_data)
            model_choice = st.sidebar.selectbox("Choose a model", list(self.models.keys()))
            model = self.models[model_choice]
            prediction = model.predict(new_data_scaled)[0]
            st.write("The patient is predicted to have ALS." if prediction == 1 else "The patient is predicted not to have ALS.")

    def display_csv_upload(self):
        st.write("# Choose a CSV file")
        uploaded_file = st.file_uploader("Upload", type="csv")
        if uploaded_file is not None:
            new_data = pd.read_csv(uploaded_file)
            if set(self.parameters).issubset(new_data.columns):
                new_data_scaled = self.scaler.transform(new_data[self.parameters])
                model_choice = st.sidebar.selectbox("Choose a model", list(self.models.keys()))
                model = self.models[model_choice]
                predictions = model.predict(new_data_scaled)
                new_data['ALS Prediction'] = predictions
                st.write("Predictions for uploaded data:")
                st.dataframe(new_data)
                self.update_performance_df(new_data)

    def display_example_data(self):
        st.write("# View Example Patients")
        num_example_patients = st.number_input("Enter the number of example patients to view:", min_value=1, max_value=100, value=10, step=1)

        if st.button("Generate Example Data"):
            num_example_patients = int(num_example_patients)
            example_data = self.create_realistic_data(num_patients=num_example_patients)
            st.dataframe(example_data)

            st.write("## Predictions for example data")
            example_data_scaled = self.scaler.transform(example_data[self.parameters])
            model_choice = st.sidebar.selectbox("Choose a model", list(self.models.keys()))
            model = self.models[model_choice]
            predictions = model.predict(example_data_scaled)
            example_data['ALS Prediction'] = predictions
            st.dataframe(example_data)

    def display_model_information(self):
        st.write("# Model Performance Comparison")

        if 'accuracy' not in self.performance_df.columns:
            st.write("Error: 'Accuracy' column not found in performance_df.")
            st.write("Available columns:", self.performance_df.columns.tolist())
            return

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
            fig = px.bar(self.performance_df, x="Model", y=metrics_to_plot, barmode="group")
            st.plotly_chart(fig)

        st.write("### Additional Model Performance Comparison")
        additional_metrics_to_plot = st.multiselect("Select additional metrics to plot", ["mcc", "balanced_accuracy", "kappa", "brier", "logloss", "f2", "jaccard", "hamming"], default=[])
        if additional_metrics_to_plot:
            fig = px.bar(self.performance_df, x="Model", y=additional_metrics_to_plot, barmode="group")
            st.plotly_chart(fig)

        if st.sidebar.checkbox("Show Metric Descriptions"):
            self.display_metric_descriptions()

    def display_graphs(self):
        st.write("# Graphs")
        st.sidebar.header("Graph Options")
        graph_options = st.sidebar.multiselect("Select Graphs", ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Feature Importance"], default=[])
        selected_model = st.sidebar.selectbox("Select Model for Graphs", list(self.models.keys()))
        show_all_models = st.sidebar.button("Show All Models for Selected Graphs")

        if st.sidebar.checkbox("Show Graph Descriptions"):
            self.display_graph_descriptions()

        if show_all_models:
            for model_name, metrics in self.model_performance.items():
                self.display_selected_graphs(metrics, model_name, graph_options)
        else:
            metrics = self.model_performance[selected_model]
            self.display_selected_graphs(metrics, selected_model, graph_options)

    def display_selected_graphs(self, metrics, model_name, graph_options):
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

    @staticmethod
    def display_metric_descriptions():
        st.write("## Metric Descriptions")

        st.write("### Accuracy")
        st.write("Accuracy is the ratio of correctly predicted instances to the total instances.")

        st.write("### Precision")
        st.write("Precision is the ratio of correctly predicted positive observations to the total predicted positives.")

        st.write("### Recall")
        st.write("Recall is the ratio of correctly predicted positive observations to all observations in actual class.")

        st.write("### F1 Score")
        st.write("The F1 Score is the weighted average of Precision and Recall.")

        st.write("### ROC AUC")
        st.write("ROC AUC is the Area Under the Receiver Operating Characteristic Curve. It represents the model's ability to distinguish between classes.")

        st.write("### MCC")
        st.write("The Matthews Correlation Coefficient (MCC) is a measure of the quality of binary classifications.")

        st.write("### Balanced Accuracy")
        st.write("Balanced Accuracy is the average of recall obtained on each class.")

        st.write("### Cohen's Kappa")
        st.write("Cohen's Kappa measures the agreement between two raters who each classify N items into C mutually exclusive categories.")

        st.write("### Brier Score")
        st.write("The Brier Score measures the mean squared difference between predicted probability and the actual outcome.")

        st.write("### Logarithmic Loss")
        st.write("Logarithmic Loss, or Log Loss, measures the performance of a classification model where the prediction input is a probability value between 0 and 1.")

        st.write("### F2 Score")
        st.write("The F2 Score is a weighted average of Precision and Recall, but with more weight given to Recall.")

        st.write("### Jaccard Index")
        st.write("The Jaccard Index measures similarity between finite sample sets.")

        st.write("### Hamming Loss")
        st.write("The Hamming Loss is the fraction of labels that are incorrectly predicted.")

    @staticmethod
    def display_graph_descriptions():
        st.write("## Graph Descriptions")

        st.write("### Confusion Matrix")
        st.write("A Confusion Matrix is a table used to evaluate the performance of a classification model. It shows the actual versus predicted classifications and is useful for understanding the number of true positives, true negatives, false positives, and false negatives.")

        st.write("### ROC Curve")
        st.write("The ROC (Receiver Operating Characteristic) Curve is a graphical representation of a model's diagnostic ability. It plots the true positive rate (sensitivity) against the false positive rate (1 - specificity) at various threshold settings. The area under the curve (AUC) represents the model's ability to distinguish between classes.")

        st.write("### Precision-Recall Curve")
        st.write("The Precision-Recall Curve is a plot that shows the trade-off between precision and recall for different threshold values. Precision is the ratio of true positive predictions to the total predicted positives, while recall is the ratio of true positive predictions to all actual positives. This curve is particularly useful for imbalanced datasets.")

        st.write("### Feature Importance")
        st.write("Feature Importance indicates the contribution of each feature to the prediction made by the model. It helps in understanding which features are most influential in the model's decision-making process. This graph is typically available for tree-based models like Random Forest and Gradient Boosting.")

if __name__ == "__main__":
    app = ALSDetectionApp()
    app.load_data()
    X_train, X_test, y_train, y_test = app.preprocess_data()
    app.train_models(X_train, y_train, X_test, y_test)
    app.run()
