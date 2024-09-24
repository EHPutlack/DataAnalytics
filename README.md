# Mitosense ALS Predictor

This app allows users to input information about patient, or patients, that may or may not have ALS. The app then uses various Machine Learning Models to determine the likelihood of ALS.

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
This version adds hyperparameter tuning, and attempts to create a proper pipeline for the app in an attempt to allow people to properly upload their own data
