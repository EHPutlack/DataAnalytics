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

class DataHandler:
    def __init__(self):
        self.parameters = self.define_parameters()
        self.scaler = StandardScaler()
    
    def define_parameters(self):
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

        return general_parameters + als_specific_parameters
    
    @st.cache_data
    def create_realistic_data(_self, num_patients=1000):
        np.random.seed(0)
        data = np.column_stack([
            np.random.normal(70, 10, num_patients),        
            np.random.normal(120, 15, num_patients),       
            np.random.normal(80, 10, num_patients),        
            np.random.normal(16, 2, num_patients),         
            np.random.normal(98, 2, num_patients),         
            np.random.normal(36.6, 0.5, num_patients),     
            np.random.normal(70, 15, num_patients),        
            np.random.normal(1.7, 0.1, num_patients),      
            np.random.normal(25, 5, num_patients),         
            np.random.normal(100, 15, num_patients),       
            np.random.normal(200, 30, num_patients),       
            np.random.normal(50, 10, num_patients),        
            np.random.normal(100, 20, num_patients),       
            np.random.normal(150, 30, num_patients),       
            np.random.normal(13.5, 1.5, num_patients),     
            np.random.normal(40, 5, num_patients),         
            np.random.normal(7000, 1500, num_patients),    
            np.random.normal(5, 0.5, num_patients),        
            np.random.normal(250000, 50000, num_patients), 
            np.random.normal(1, 0.2, num_patients),        
            np.random.normal(15, 5, num_patients),         
            np.random.normal(140, 5, num_patients),        
            np.random.normal(4, 0.5, num_patients),        
            np.random.normal(9.5, 0.5, num_patients),      
            np.random.normal(2, 0.2, num_patients),        
            np.random.normal(50, 10, num_patients),        
            np.random.normal(30, 5, num_patients),         
            np.random.normal(60, 10, num_patients),        
            np.random.normal(40, 10, num_patients),        
            np.random.normal(30, 10, num_patients),        
        ])

        half_patients = num_patients // 2
        labels = np.concatenate([np.ones(half_patients), np.zeros(num_patients - half_patients)])
        df = pd.DataFrame(data, columns=_self.parameters)
        df['ALS'] = labels
        return df
    
    def load_data(self):
        uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
        else:
            df = self.create_realistic_data()
        return df

class ModelHandler:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.models = self.define_models()
        self.performance_df = pd.DataFrame()

    def define_models(self):
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
    
    def preprocess_data(self, df):
        X = df.drop(columns=['ALS'])
        y = df['ALS']
        X_scaled = self.data_handler.scaler.fit_transform(X)
        return train_test_split(X_scaled, y, test_size=0.2, random_state=0)

    def evaluate_models(self, X_train, X_test, y_train, y_test):
        performance_metrics = []
        model_performance = {}

        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test)

            metrics = {
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

            model_performance[model_name] = {"model": model, **metrics}
            performance_metrics.append({"Model": model_name, **metrics})

        self.performance_df = pd.DataFrame(performance_metrics)
        return model_performance, self.performance_df

class ReportHandler:
    def __init__(self, model_performance, performance_df):
        self.model_performance = model_performance
        self.performance_df = performance_df
    
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
                fig, ax = plt.subplots(figsize=(8, 6))  
                sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"Confusion Matrix for {model_name}")
                temp_image_path = f"{model_name}_confusion_matrix.png"
                fig.savefig(temp_image_path, bbox_inches='tight')
                pdf.add_page()
                pdf.cell(200, 10, txt=f"Confusion Matrix for {model_name}", ln=True, align="L")
                pdf.image(temp_image_path, w=180)
                temp_images.append(temp_image_path)

            if "ROC Curve" in graph_options:
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
                pdf.add_page()
                pdf.cell(200, 10, txt=f"ROC Curve for {model_name}", ln=True, align="L")
                pdf.image(temp_image_path, w=180)
                temp_images.append(temp_image_path)

            if "Precision-Recall Curve" in graph_options:
                fig, ax = plt.subplots(figsize=(8, 6))  
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
                pdf.image(temp_image_path, w=180)
                temp_images.append(temp_image_path)

            if "Feature Importance" in graph_options and hasattr(metrics["model"], "feature_importances_"):
                feature_importance = pd.DataFrame({
                    'Feature': data_handler.parameters,
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
            self.performance_df.plot(kind="bar", x="Model", y=["accuracy", "precision", "recall", "f1", "roc_auc"], ax=ax)
            ax.set_title("Model Performance Comparison")
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.6), ncol=2)
            temp_image_path = "model_performance_comparison.png"
            fig.savefig(temp_image_path, bbox_inches='tight')
            pdf.add_page()
            pdf.cell(200, 10, txt="Model Performance Comparison", ln=True, align="L")
            pdf.image(temp_image_path, w=180)
            temp_images.append(temp_image_path)

            fig, ax = plt.subplots(figsize=(8, 6))
            self.performance_df.plot(kind="bar", x="Model", y=["mcc", "balanced_accuracy", "kappa", "brier", "logloss", "f2", "jaccard", "hamming"], ax=ax)
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

def main():
    st.title("ALS Detection Model")

    data_handler = DataHandler()
    model_handler = ModelHandler(data_handler)

    # Load data
    df = data_handler.load_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = model_handler.preprocess_data(df)

    # Evaluate models
    model_performance, performance_df = model_handler.evaluate_models(X_train, X_test, y_train, y_test)

    # Display model performance
    st.write("# Model Performance Comparison")
    st.dataframe(performance_df)

    best_model = performance_df.loc[performance_df["accuracy"].idxmax()]
    st.write(f"### Best Model: {best_model['Model']}")
    st.write(f"Accuracy: {best_model['accuracy']:.2f}")
    st.write(f"Precision: {best_model['precision']:.2f}")
    st.write(f"Recall: {best_model['recall']:.2f}")
    st.write(f"F1 Score: {best_model['f1']:.2f}")
    st.write(f"ROC AUC: {best_model['roc_auc']:.2f}")

    st.write("### Plotting the Model Performance Comparison")
    metrics_to_plot = st.multiselect("Select metrics to plot", ["accuracy", "precision", "recall", "f1", "roc_auc"], default=[])
    if metrics_to_plot:
        fig = px.bar(performance_df, x="Model", y=metrics_to_plot, barmode="group")
        st.plotly_chart(fig)

    if st.sidebar.button("Save Report to PDF"):
        report_handler = ReportHandler(model_performance, performance_df)
        report_handler.save_report_to_pdf()

if __name__ == "__main__":
    main()
