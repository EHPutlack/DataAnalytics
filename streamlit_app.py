import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Set up the page title and description
st.set_page_config(page_title="ALS Detection", page_icon="🩺")
st.title("🩺 ALS Detection App")
st.write(
    """
    This app simulates data for 1,000 patients to compare 25 physiological parameters.
    50% of the patients have ALS. Enter new patient data to determine if they have ALS.
    """
)

# Function to create realistic fake data
@st.cache_data
def create_realistic_data(num_patients=1000):
    np.random.seed(0)
    parameters = [
        'Heart Rate', 'Blood Pressure Systolic', 'Blood Pressure Diastolic', 
        'Respiratory Rate', 'Oxygen Saturation', 'Temperature', 'Weight', 
        'Height', 'BMI', 'Blood Glucose', 'Cholesterol', 'HDL', 'LDL', 
        'Triglycerides', 'Hemoglobin', 'Hematocrit', 'WBC Count', 
        'RBC Count', 'Platelet Count', 'Creatinine', 'BUN', 'Sodium', 
        'Potassium', 'Calcium', 'Magnesium'
    ]
    # Generating realistic data distributions
    data = np.column_stack([
        np.random.normal(70, 10, num_patients),       # Heart Rate
        np.random.normal(120, 15, num_patients),      # Blood Pressure Systolic
        np.random.normal(80, 10, num_patients),       # Blood Pressure Diastolic
        np.random.normal(16, 2, num_patients),        # Respiratory Rate
        np.random.normal(98, 2, num_patients),        # Oxygen Saturation
        np.random.normal(36.6, 0.5, num_patients),    # Temperature
        np.random.normal(70, 15, num_patients),       # Weight
        np.random.normal(1.7, 0.1, num_patients),     # Height
        np.random.normal(25, 5, num_patients),        # BMI
        np.random.normal(100, 15, num_patients),      # Blood Glucose
        np.random.normal(200, 30, num_patients),      # Cholesterol
        np.random.normal(50, 10, num_patients),       # HDL
        np.random.normal(100, 20, num_patients),      # LDL
        np.random.normal(150, 30, num_patients),      # Triglycerides
        np.random.normal(13.5, 1.5, num_patients),    # Hemoglobin
        np.random.normal(40, 5, num_patients),        # Hematocrit
        np.random.normal(7000, 1500, num_patients),   # WBC Count
        np.random.normal(5, 0.5, num_patients),       # RBC Count
        np.random.normal(250000, 50000, num_patients),# Platelet Count
        np.random.normal(1, 0.2, num_patients),       # Creatinine
        np.random.normal(15, 5, num_patients),        # BUN
        np.random.normal(140, 5, num_patients),       # Sodium
        np.random.normal(4, 0.5, num_patients),       # Potassium
        np.random.normal(9.5, 0.5, num_patients),     # Calcium
        np.random.normal(2, 0.2, num_patients)        # Magnesium
    ])
    labels = np.random.choice([0, 1], size=(num_patients,), p=[0.5, 0.5])
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

# Train a Random Forest model
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
st.write(f"Cross-validated accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Display model accuracy on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model accuracy on test set: {accuracy:.2f}")

# User input for new patient data
st.write("## Enter new patient data")
new_data = []
parameters = [
    'Heart Rate', 'Blood Pressure Systolic', 'Blood Pressure Diastolic', 
    'Respiratory Rate', 'Oxygen Saturation', 'Temperature', 'Weight', 
    'Height', 'BMI', 'Blood Glucose', 'Cholesterol', 'HDL', 'LDL', 
    'Triglycerides', 'Hemoglobin', 'Hematocrit', 'WBC Count', 
    'RBC Count', 'Platelet Count', 'Creatinine', 'BUN', 'Sodium', 
    'Potassium', 'Calcium', 'Magnesium'
]

for param in parameters:
    value = st.number_input(f"{param}", min_value=0.0, step=0.1, format="%.1f")
    new_data.append(value)

# Predict ALS for the new patient data
if st.button("Predict ALS"):
    new_data = np.array(new_data).reshape(1, -1)
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)[0]
    if prediction == 1:
        st.write("The patient is predicted to have ALS.")
    else:
        st.write("The patient is predicted not to have ALS.")
