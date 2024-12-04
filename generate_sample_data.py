import pandas as pd
import numpy as np

def create_sample_dataset(num_samples=100):
    np.random.seed(42)  # For reproducibility
    
    data = {
        'Heart Rate': np.random.normal(75, 12, num_samples),  # 60-100 bpm
        'Blood Pressure Systolic': np.random.normal(120, 10, num_samples),  # 90-140 mmHg
        'Blood Pressure Diastolic': np.random.normal(80, 8, num_samples),  # 60-90 mmHg
        'Respiratory Rate': np.random.normal(16, 2, num_samples),  # 12-20 breaths/min
        'Oxygen Saturation': np.random.normal(97, 2, num_samples),  # 95-100%
        'Temperature': np.random.normal(37, 0.5, num_samples),  # 36.5-37.5°C
        'Weight': np.random.normal(70, 15, num_samples),  # 45-100 kg
        'Height': np.random.normal(170, 10, num_samples),  # 150-190 cm
        'BMI': np.random.normal(25, 4, num_samples),  # 18.5-30
        'Blood Glucose': np.random.normal(100, 20, num_samples),  # 70-140 mg/dL
        'Cholesterol': np.random.normal(190, 30, num_samples),  # 150-240 mg/dL
        'HDL': np.random.normal(50, 10, num_samples),  # 40-60 mg/dL
        'LDL': np.random.normal(100, 20, num_samples),  # 70-130 mg/dL
        'Triglycerides': np.random.normal(150, 30, num_samples),  # 100-200 mg/dL
        'Hemoglobin': np.random.normal(14, 1.5, num_samples),  # 12-17 g/dL
        'Hematocrit': np.random.normal(42, 4, num_samples),  # 36-48%
        'WBC Count': np.random.normal(7500, 1500, num_samples),  # 4500-11000/µL
        'RBC Count': np.random.normal(4.8, 0.4, num_samples),  # 4.2-5.4 million/µL
        'Platelet Count': np.random.normal(250000, 50000, num_samples),  # 150000-450000/µL
        'Creatinine': np.random.normal(1.0, 0.2, num_samples),  # 0.6-1.2 mg/dL
        'BUN': np.random.normal(15, 4, num_samples),  # 7-20 mg/dL
        'Sodium': np.random.normal(140, 2, num_samples),  # 135-145 mEq/L
        'Potassium': np.random.normal(4.0, 0.3, num_samples),  # 3.5-5.0 mEq/L
        'Calcium': np.random.normal(9.5, 0.5, num_samples),  # 8.5-10.5 mg/dL
        'Magnesium': np.random.normal(2.0, 0.2, num_samples),  # 1.7-2.2 mg/dL
        'Muscle Strength': np.random.normal(80, 10, num_samples),  # 0-100 scale
        'Motor Function Score': np.random.normal(85, 10, num_samples),  # 0-100 scale
        'Speech Clarity': np.random.normal(90, 8, num_samples),  # 0-100 scale
        'Swallowing Function': np.random.normal(85, 10, num_samples),  # 0-100 scale
        'Respiratory Capacity': np.random.normal(80, 12, num_samples)  # 0-100 scale
    }
    
    df = pd.DataFrame(data)
    
    # Ensure all values are within realistic ranges
    df = df.clip(lower=0)  # No negative values
    
    # Save to both CSV and Excel formats
    df.to_csv('sample_patient_data.csv', index=False)
    df.to_excel('sample_patient_data.xlsx', index=False)
    
    return df

# Create the sample dataset
df = create_sample_dataset(100)
print("Files 'sample_patient_data.csv' and 'sample_patient_data.xlsx' have been created.")
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())