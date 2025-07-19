import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Title of the app
st.title("Employee Salary Prediction App")
st.write("This app predicts the Salary based on Experience using Machine Learning.")

# Upload CSV dataset
uploaded_file = st.file_uploader("Upload your CSV file (with 'YearsExperience' and 'Salary')", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Prepare data
    X = df[['YearsExperience']]
    y = df['Salary']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Display results
    st.subheader("Model Evaluation")
    st.write("R2 Score: ", r2_score(y_test, y_pred))
    st.write("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

    # Input for prediction
    exp = st.number_input("Enter Years of Experience", min_value=0.0, step=0.1)
    if st.button("Predict Salary"):
        salary = model.predict(np.array([[exp]]))
        st.success(f"Predicted Salary: ₹ {salary[0]:,.2f}")
