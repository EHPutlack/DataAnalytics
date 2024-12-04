from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

app = Flask(__name__)
app.secret_key = "some_secret_key"  # For flash messages


class ALSDetectionApp:
    def __init__(self):
        self.general_parameters = [
            'Heart Rate', 'Blood Pressure Systolic', 'Blood Pressure Diastolic', 'Respiratory Rate', 'Oxygen Saturation',
            'Temperature', 'Weight', 'Height', 'BMI', 'Blood Glucose', 'Cholesterol', 'HDL', 'LDL', 'Triglycerides',
            'Hemoglobin', 'Hematocrit', 'WBC Count', 'RBC Count', 'Platelet Count', 'Creatinine', 'BUN', 'Sodium',
            'Potassium', 'Calcium', 'Magnesium'
        ]
        self.als_specific_parameters = [
            'Muscle Strength', 'Motor Function Score', 'Speech Clarity', 'Swallowing Function', 'Respiratory Capacity'
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

    def create_realistic_data(self, parameters, num_patients=1000):
        np.random.seed(0)
        data = np.column_stack([
            np.random.normal(70, 10, num_patients),  # Heart Rate
            np.random.normal(120, 15, num_patients),  # Blood Pressure Systolic
            np.random.normal(80, 10, num_patients),  # Blood Pressure Diastolic
            np.random.normal(16, 2, num_patients),  # Respiratory Rate
            np.random.normal(98, 2, num_patients),  # Oxygen Saturation
            np.random.normal(36.6, 0.5, num_patients),  # Temperature
            np.random.normal(70, 15, num_patients),  # Weight
            np.random.normal(1.7, 0.1, num_patients),  # Height
            np.random.normal(25, 5, num_patients),  # BMI
            np.random.normal(100, 15, num_patients),  # Blood Glucose
            np.random.normal(200, 30, num_patients),  # Cholesterol
            np.random.normal(50, 10, num_patients),  # HDL
            np.random.normal(100, 20, num_patients),  # LDL
            np.random.normal(150, 30, num_patients),  # Triglycerides
            np.random.normal(13.5, 1.5, num_patients),  # Hemoglobin
            np.random.normal(40, 5, num_patients),  # Hematocrit
            np.random.normal(7000, 1500, num_patients),  # WBC Count
            np.random.normal(5, 0.5, num_patients),  # RBC Count
            np.random.normal(250000, 50000, num_patients),  # Platelet Count
            np.random.normal(1, 0.2, num_patients),  # Creatinine
            np.random.normal(15, 5, num_patients),  # BUN
            np.random.normal(140, 5, num_patients),  # Sodium
            np.random.normal(4, 0.5, num_patients),  # Potassium
            np.random.normal(9.5, 0.5, num_patients),  # Calcium
            np.random.normal(2, 0.2, num_patients),  # Magnesium
            np.random.normal(50, 10, num_patients),  # Muscle Strength
            np.random.normal(30, 5, num_patients),  # Motor Function Score
            np.random.normal(60, 10, num_patients),  # Speech Clarity
            np.random.normal(40, 10, num_patients),  # Swallowing Function
            np.random.normal(30, 10, num_patients),  # Respiratory Capacity
        ])

        half_patients = num_patients // 2
        labels = np.concatenate([np.ones(half_patients), np.zeros(num_patients - half_patients)])
        df = pd.DataFrame(data, columns=parameters)
        df['ALS'] = labels
        return df

    def load_data(self):
        self.df = self.create_realistic_data(self.parameters)

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
            })

        self.performance_df = pd.DataFrame(performance_metrics)


# Initialize the ALSDetectionApp
als_app = ALSDetectionApp()
als_app.load_data()
X_train, X_test, y_train, y_test = als_app.preprocess_data()
als_app.train_models(X_train, y_train, X_test, y_test)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model_information')
def model_information():
    # Pass performance metrics and model names to the HTML template
    return render_template('model_information.html', performance=als_app.performance_df)


@app.route('/graph/<model_name>/<graph_type>')
def graph(model_name, graph_type):
    metrics = als_app.model_performance.get(model_name)
    if not metrics:
        return "Model not found", 404

    img = io.BytesIO()

    if graph_type == 'confusion_matrix':
        # Plot confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f'Confusion Matrix for {model_name}')
    elif graph_type == 'roc_curve':
        # Plot ROC curve
        fpr, tpr, _ = metrics['roc_curve']
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["roc_auc"]:.2f})')
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(f'ROC Curve for {model_name}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
    elif graph_type == 'precision_recall_curve':
        # Plot Precision-Recall curve
        precision, recall, _ = metrics['precision_recall_curve']
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label=f'{model_name}')
        ax.set_title(f'Precision-Recall Curve for {model_name}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

    fig.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    return send_file(img, mimetype='image/png')


@app.route('/display_graphs/<model_name>')
def display_graphs(model_name):
    # Dynamically display all graphs for a specific model
    return render_template('display_graphs.html', model_name=model_name)


if __name__ == '__main__':
    app.run(debug=True)
