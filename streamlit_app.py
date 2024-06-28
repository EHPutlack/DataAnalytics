import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Function to create fake data
@st.cache_data
def create_fake_data(num_patients=1000, num_parameters=25):
    np.random.seed(0)
    data = np.random.rand(num_patients, num_parameters)
    labels = np.random.choice([0, 1], size=(num_patients,), p=[0.5, 0.5])
    df = pd.DataFrame(data, columns=[f'param_{i+1}' for i in range(num_parameters)])
    df['ALS'] = labels
    return df

# Load and display the fake data
df = create_fake_data()
st.dataframe(df.head())

# Split the data into training and test sets
X = df.drop(columns=['ALS'])
y = df['ALS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Display model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model accuracy: {accuracy:.2f}")

# User input for new patient data
st.write("## Enter new patient data")
new_data = []
for i in range(25):
    value = st.number_input(f"Parameter {i+1}", min_value=0.0, max_value=1.0, value=0.5)
    new_data.append(value)

# Predict ALS for the new patient data
if st.button("Predict ALS"):
    new_data = np.array(new_data).reshape(1, -1)
    prediction = model.predict(new_data)[0]
    if prediction == 1:
        st.write("The patient is predicted to have ALS.")
    else:
        st.write("The patient is predicted not to have ALS.")

