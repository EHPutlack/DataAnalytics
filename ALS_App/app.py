from flask import Flask, render_template, request, send_file, redirect, url_for, flash, session, make_response, Response
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import logging
import threading
import matplotlib.colors as mcolors
logging.basicConfig(level=logging.INFO)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, matthews_corrcoef, balanced_accuracy_score, cohen_kappa_score, brier_score_loss, log_loss, fbeta_score, jaccard_score, hamming_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from flask_caching import Cache
import base64
import os
import psycopg2
from dotenv import load_dotenv
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
from datetime import datetime

load_dotenv()  # load environment variables

if 'DATABASE_URL' not in os.environ:
    raise Exception("DATABASE_URL not found in environment variables")

try:
    # Database connection pool
    db_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=os.environ['DATABASE_URL']
    )
except Exception as e:
    print(f"Failed to create connection pool: {str(e)}")
    db_pool = None

@contextmanager
def get_db_connection():
    try:
        connection = db_pool.getconn()
        yield connection
    finally:
        db_pool.putconn(connection)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_format(filename):
    """
    Validate if the uploaded file has an allowed extension
    Returns: tuple (is_valid: bool, extension: str)
    """
    if '.' not in filename:
        return False, None
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS, extension
    
def read_data_file(file, file_extension):
    """
    Read data from uploaded file based on extension
    Returns: tuple (success: bool, data: DataFrame or None, error_message: str or None)
    """
    try:
        if file_extension == 'csv':
            df = pd.read_csv(file.stream)
        elif file_extension == 'xlsx':
            df = pd.read_excel(file.stream)
        return True, df, None
    except Exception as e:
        return False, None, f"Error reading file: {str(e)}"
        
def validate_required_columns(df):
    """
    Check if all required columns are present in the dataset
    Returns: tuple (is_valid: bool, missing_columns: list)
    """
    required_columns = [
        'Heart Rate', 'Blood Pressure Systolic', 'Blood Pressure Diastolic',
        'Respiratory Rate', 'Oxygen Saturation', 'Temperature', 'Weight',
        'Height', 'BMI', 'Blood Glucose', 'Cholesterol', 'HDL', 'LDL',
        'Triglycerides', 'Hemoglobin', 'Hematocrit', 'WBC Count',
        'RBC Count', 'Platelet Count', 'Creatinine', 'BUN', 'Sodium',
        'Potassium', 'Calcium', 'Magnesium', 'Muscle Strength',
        'Motor Function Score', 'Speech Clarity', 'Swallowing Function',
        'Respiratory Capacity'
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns
    
def calculate_dataset_statistics(df):
    """
    Calculate comprehensive statistics for the dataset
    Returns: dict containing various statistics
    """
    total_records = len(df)
    stats = {
        'total_records': total_records,
        'missing_values': df.isnull().sum().to_dict(),
        'column_stats': {}
    }
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        stats['column_stats'][col] = {
            'mean': float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
            'std': float(df[col].std()) if pd.api.types.is_numeric_dtype(df[col]) else None,
            'min': float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
            'max': float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
            'missing_count': int(missing_count),
            'missing_percentage': float((missing_count / total_records) * 100)
        }
    
    return stats
    
def save_to_database(df, table_name):
    """
    Save the DataFrame to PostgreSQL database and maintain a backup
    Returns: tuple (success: bool, error_message: str or None)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create main table
                columns_def = ', '.join([
                    f"{col.lower().replace(' ', '_')} FLOAT" for col in df.columns
                ])
                base_table_structure = f"""
                    id SERIAL PRIMARY KEY,
                    {columns_def},
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                """
                
                # Create main table
                create_table_query = f"""
                CREATE TABLE {table_name} (
                    {base_table_structure}
                )
                """
                cur.execute(create_table_query)
                
                # Create backup table with the same structure
                backup_table_name = f"{table_name}_backup"
                create_backup_table_query = f"""
                CREATE TABLE {backup_table_name} (
                    {base_table_structure}
                )
                """
                cur.execute(create_backup_table_query)
                
                # Insert data into both tables
                for _, row in df.iterrows():
                    columns = ', '.join([col.lower().replace(' ', '_') for col in df.columns])
                    placeholders = ', '.join(['%s'] * len(df.columns))
                    
                    # Insert into main table
                    insert_query = f"""
                    INSERT INTO {table_name} ({columns})
                    VALUES ({placeholders})
                    """
                    cur.execute(insert_query, list(row))
                    
                    # Insert into backup table
                    backup_insert_query = f"""
                    INSERT INTO {backup_table_name} ({columns})
                    VALUES ({placeholders})
                    """
                    cur.execute(backup_insert_query, list(row))
                
                conn.commit()
                return True, None
    except Exception as e:
        return False, f"Database error: {str(e)}"
        
def process_and_validate_upload(file):
    """
    Main function to process and validate file upload
    Returns: dict containing processing results and statistics
    """
    # Validate file format
    is_valid_format, file_extension = validate_file_format(file.filename)
    if not is_valid_format:
        return {
            'success': False,
            'message': 'Invalid file format. Please upload CSV or XLSX file.',
            'stats': None
        }

    # Read file
    read_success, df, read_error = read_data_file(file, file_extension)
    if not read_success:
        return {
            'success': False,
            'message': read_error,
            'stats': None
        }

    # Validate columns
    columns_valid, missing_columns = validate_required_columns(df)
    if not columns_valid:
        return {
            'success': False,
            'message': f'Missing required columns: {", ".join(missing_columns)}',
            'stats': None
        }

    # Calculate detailed statistics
    stats = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'column_stats': {
            column: {
                'mean': float(df[column].mean()) if pd.api.types.is_numeric_dtype(df[column]) else None,
                'std': float(df[column].std()) if pd.api.types.is_numeric_dtype(df[column]) else None,
                'min': float(df[column].min()) if pd.api.types.is_numeric_dtype(df[column]) else None,
                'max': float(df[column].max()) if pd.api.types.is_numeric_dtype(df[column]) else None
            } for column in df.columns
        }
    }

    # Generate unique table name
    table_name = f"patient_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    # Save to database
    db_success, db_error = save_to_database(df, table_name)
    if not db_success:
        return {
            'success': False,
            'message': db_error,
            'stats': None
        }
        
    if df.empty:
        return {
            'success': False,
            'message': 'The uploaded file is empty.',
            'stats': None
        }
        
    return {
        'success': True,
        'message': f'Successfully uploaded {len(df)} records to database.',
        'stats': stats,
        'table_name': table_name
    }

def clean_database(preserve_backup=True):
    """
    Clean database tables with option to preserve backup tables
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # First, disable foreign key constraints
                cur.execute("SET CONSTRAINTS ALL DEFERRED;")
                
                # Get all tables in the current schema
                cur.execute("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = current_schema()
                    AND (NOT %s OR tablename NOT LIKE '%%_backup')
                """, (preserve_backup,))
                
                tables = cur.fetchall()
                
                # Drop each table
                for (table,) in tables:
                    cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
                
                # Clean all sequences
                cur.execute("""
                    SELECT sequence_name 
                    FROM information_schema.sequences 
                    WHERE sequence_schema = current_schema()
                """)
                
                sequences = cur.fetchall()
                for (sequence,) in sequences:
                    cur.execute(f'DROP SEQUENCE IF EXISTS "{sequence}" CASCADE')
                
                # Clean all views
                cur.execute("""
                    SELECT viewname 
                    FROM pg_views 
                    WHERE schemaname = current_schema()
                """)
                
                views = cur.fetchall()
                for (view,) in views:
                    cur.execute(f'DROP VIEW IF EXISTS "{view}" CASCADE')
                
                conn.commit()
        return True, None
    except Exception as e:
        return False, f"Error cleaning database: {str(e)}"

def restore_from_backup(table_name):
    """
    Restore data from backup table
    Returns: tuple (success: bool, error_message: str or None)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                backup_table_name = f"{table_name}_backup"

                # Check if backup exists
                check_backup_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
                """
                cur.execute(check_backup_query, (backup_table_name,))
                backup_exists = cur.fetchone()[0]

                if not backup_exists:
                    return False, "No backup table found"

                # Clear main table
                cur.execute(f"TRUNCATE TABLE {table_name}")

                # Copy data from backup
                restore_query = f"""
                INSERT INTO {table_name}
                SELECT * FROM {backup_table_name}
                """
                cur.execute(restore_query)

                conn.commit()
                return True, None
    except Exception as e:
        return False, f"Restore error: {str(e)}"

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(24)

cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(app)

graph_lock = threading.Lock()

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
            np.random.normal(30, 10, num_patients),  # Respiratory Capacity,
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
                "precision_recall_curve": precision_recall_curve(y_test, y_prob),
                "mcc": matthews_corrcoef(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "kappa": cohen_kappa_score(y_test, y_pred),
                "brier": brier_score_loss(y_test, y_prob),
                "logloss": log_loss(y_test, y_prob),
                "f2": fbeta_score(y_test, y_pred, beta=2),
                "jaccard": jaccard_score(y_test, y_pred),
                "hamming": hamming_loss(y_test, y_pred)
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
    return render_template('model_information.html', performance=als_app.performance_df)

@app.route('/download_plot_pdf')
def download_plot_pdf():
    # Get the current figure
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='pdf')
    img_buf.seek(0)
    
    return send_file(
        img_buf,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='metrics_plot.pdf'
    )

@app.route('/data_input', methods=['GET', 'POST'])
def data_input():
    if request.method == 'POST':
        print("File upload attempted")
        if 'file' not in request.files:
            return render_template('data_input.html', 
                                 error='No file uploaded')
        
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Process in chunks for large files
            chunk_size = 8192  # 8KB chunks
            file_content = bytearray()
            while True:
                chunk = file.stream.read(chunk_size)
                if not chunk:
                    break
                file_content.extend(chunk)
        if file.filename == '':
            return render_template('data_input.html', 
                                 error='No file selected')

        # Debug print
        print(f"File received: {file.filename}")

        # Get the selected model
        selected_model = request.form.get('model', '')
        if not selected_model:
            return render_template('data_input.html', 
                                 error='No model selected')

        # Process the upload and get detailed statistics
        try:
            result = process_and_validate_upload(file)
            if not result['success']:
                return render_template('data_input.html', 
                                     error=result['message'])

            # Store stats in session for upload_success page
            session['upload_stats'] = result['stats']
            session['table_name'] = result['table_name']

            # Calculate additional statistics for display
            display_stats = {
                'total_records': result['stats']['total_records'],
                'missing_values': result['stats']['missing_values'],
                'column_statistics': {}
            }

            # Format column statistics for display
            for col, col_stats in result['stats']['column_stats'].items():
                if col_stats['mean'] is not None:
                    missing_count = col_stats.get('missing_count', 0)
                    total_records = result['stats']['total_records']
                    missing_percentage = (missing_count / total_records * 100) if total_records > 0 else 0
                    
                    display_stats['column_statistics'][col] = {
                        'mean': f"{col_stats['mean']:.2f}",
                        'std': f"{col_stats['std']:.2f}",
                        'min': f"{col_stats['min']:.2f}",
                        'max': f"{col_stats['max']:.2f}",
                        'missing_percentage': f"{missing_percentage:.1f}%"
                    }

            return render_template('data_input.html', 
                                 success_message=f"Successfully processed {display_stats['total_records']} records",
                                 stats=display_stats,
                                 table_name=result.get('table_name'))

        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('data_input'))
    
    # GET request - show upload form
    return render_template('data_input.html')

@app.route('/plot_metrics')
def plot_metrics():
    selected_metrics = request.args.getlist('metrics')

    if not selected_metrics:
        return "No metrics selected", 400

    # Create a larger figure with more space for labels and legend
    fig, ax = plt.subplots(figsize=(15, 8))  # Increased figure size

    # Plot the data
    als_app.performance_df.set_index('Model')[selected_metrics].plot(
        kind='bar', 
        ax=ax, 
        color=list(mcolors.TABLEAU_COLORS.values()), 
        edgecolor='black'
    )

    # Customize the plot
    ax.set_title("Model Performance Comparison", pad=20)
    ax.set_xlabel("Model", labelpad=10)
    ax.set_ylabel("Scores", labelpad=10)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust legend position and layout
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot to BytesIO
    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    plt.close(fig)

    img.seek(0)
    base64_img = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template('plot_metrics.html', plot_image=base64_img)

@app.route('/display_graphs/<model_name>')
def display_graphs(model_name):
    return render_template('display_graphs.html', model_name=model_name)


@app.route('/graph/<model_name>/<graph_type>')
def graph(model_name, graph_type):
    # Acquire the lock to ensure thread safety
    with graph_lock:
        model_metrics = als_app.model_performance.get(model_name)

        if not model_metrics:
            return f"No data found for model: {model_name}", 404

        fig, ax = plt.subplots(figsize=(10, 6))

        if graph_type == 'confusion_matrix':
            sns.heatmap(model_metrics['confusion_matrix'], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix for {model_name}")
            
        elif graph_type == 'roc_curve':
            fpr, tpr, _ = model_metrics['roc_curve']
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {model_metrics['roc_auc']:.2f})")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title(f"ROC Curve for {model_name}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            
        elif graph_type == 'precision_recall_curve':
            precision, recall, _ = model_metrics['precision_recall_curve']
            ax.plot(recall, precision, label=f"{model_name}")
            ax.set_title(f"Precision-Recall Curve for {model_name}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend(loc="lower left")
            
        elif graph_type == 'feature_importance':
            model = model_metrics['model']
            
            # Check for different types of feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For models like Logistic Regression that use coefficients
                importances = np.abs(model.coef_[0])  # Use absolute values for coefficient importance
            else:
                # Return a placeholder image for models without feature importance
                ax.text(0.5, 0.5, 'Feature Importance not available for this model',
                       horizontalalignment='center', verticalalignment='center')
                ax.set_xticks([])
                ax.set_yticks([])
                
            if 'importances' in locals():
                # Create feature importance plot
                feature_importance = pd.DataFrame({
                    'Feature': als_app.parameters,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True)
                
                # Create horizontal bar plot using matplotlib instead of seaborn
                feature_importance.plot(kind='barh', x='Feature', y='Importance', ax=ax)
                # Or alternatively:
                # ax.barh(y=feature_importance['Feature'], width=feature_importance['Importance'])
                
                ax.set_title(f"Feature Importance for {model_name}")
                ax.set_xlabel("Importance")
                ax.set_ylabel("Features")
                
                # Adjust layout to prevent label cutoff
                plt.tight_layout()

        else:
            return f"Graph type {graph_type} not recognized", 400

        # Save plot to an in-memory buffer with increased DPI for clarity
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')  # Added bbox_inches='tight'
        plt.close(fig)
        img.seek(0)

        response = make_response(img.read())
        response.headers['Content-Type'] = 'image/png'
        return response

@app.route('/download_graph_pdf/<model_name>/<graph_type>')
def download_graph_pdf(model_name, graph_type):
    # Create the specific graph
    fig, ax = plt.subplots(figsize=(10, 6))

    if graph_type == 'confusion_matrix':
        sns.heatmap(model_metrics[model_name]['confusion_matrix'], 
                   annot=True, fmt='d', ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}')
    elif graph_type == 'roc_curve':
        # Your existing ROC curve plotting code
        ax.plot(model_metrics[model_name]['fpr'], 
                model_metrics[model_name]['tpr'])
        ax.set_title(f'ROC Curve - {model_name}')
    elif graph_type == 'precision_recall_curve':
        # Your existing PR curve plotting code
        ax.plot(model_metrics[model_name]['recall'], 
                model_metrics[model_name]['precision'])
        ax.set_title(f'Precision-Recall Curve - {model_name}')
    elif graph_type == 'feature_importance':
        # Your existing feature importance plotting code
        feature_importance = pd.DataFrame({
            'Feature': als_app.parameters,
            'Importance': model_metrics[model_name]['feature_importance']
        }).sort_values(by='Importance', ascending=True)
        feature_importance.plot(kind='barh', x='Feature', y='Importance', ax=ax)
        ax.set_title(f'Feature Importance - {model_name}')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='pdf')
    img_buf.seek(0)
    plt.close(fig)

    return send_file(
        img_buf,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'{model_name}_{graph_type}.pdf'
    )

@app.route('/upload_success')
def upload_success():
    """Display upload statistics after successful file processing"""
    stats = session.get('upload_stats', {})
    table_name = session.get('table_name', '')
    return render_template('upload_success.html', 
                         stats=stats,
                         table_name=table_name)

@app.route('/clear_database', methods=['POST'])
def clear_database():
    try:
        success, error = clean_database(preserve_backup=True)
        if success:
            flash('Database cleared successfully', 'success')
        else:
            flash(f'Error clearing database: {error}', 'error')
    except Exception as e:
        flash(f'Error clearing database: {str(e)}', 'error')
    return redirect(url_for('data_input'))

@app.route('/restore_backup/<table_name>', methods=['POST'])
def restore_backup(table_name):
    success, error = restore_from_backup(table_name)
    if success:
        flash('Data restored from backup successfully', 'success')
    else:
        flash(f'Error restoring backup: {error}', 'error')
    return redirect(url_for('data_input'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
