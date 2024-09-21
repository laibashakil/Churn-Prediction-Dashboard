import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load the trained model and feature list
model = joblib.load('churn_prediction_model.pkl')
feature_list = joblib.load('feature_list.pkl')

# App title and styling
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")
st.title("🎯 Churn Prediction Dashboard")

# Sidebar for user input
st.sidebar.header("Enter Customer Details")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 1)
    MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0, 118.0, 70.0)
    TotalCharges = st.sidebar.slider('Total Charges', 0.0, 9000.0, 3000.0)
    InternetService = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    Contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    PaymentMethod = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))

    data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'gender_Male': 1 if gender == 'Male' else 0,
        'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
        'InternetService_No': 1 if InternetService == 'No' else 0,
        'Contract_One year': 1 if Contract == 'One year' else 0,
        'Contract_Two year': 1 if Contract == 'Two year' else 0,
        'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if PaymentMethod == 'Bank transfer (automatic)' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0
    }

    input_data = pd.DataFrame([data])

    # Ensure the input data has all the columns that the model was trained on
    missing_cols = [col for col in feature_list if col not in input_data.columns]

    # Add missing columns with default value of 0 in a single step using pd.concat
    if missing_cols:
        missing_data = pd.DataFrame(0, index=input_data.index, columns=missing_cols)
        input_data = pd.concat([input_data, missing_data], axis=1)

    # Reorder columns to match the training data
    input_data = input_data[feature_list]

    return input_data

# Get user input data
input_data = user_input_features()

# Predict churn
prediction_proba = model.predict_proba(input_data)

# Create two columns for displaying results and charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Prediction Result")
    st.write('Prediction: **Churn**' if prediction_proba[0][1] > 0.5 else 'Prediction: **No Churn**')

    st.subheader("Prediction Probabilities")
    st.write(f"Probability of Churn: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of Not Churning: {prediction_proba[0][0]:.2f}")

with col2:
    # Create a visual chart of the prediction probabilities
    st.subheader("Churn Probability Visualization")
    probabilities = pd.DataFrame({
        'Outcome': ['Churn', 'No Churn'],
        'Probability': [prediction_proba[0][1], prediction_proba[0][0]]
    })

    fig = px.bar(probabilities, x='Outcome', y='Probability', color='Outcome',
                 color_discrete_map={'Churn': '#FF6347', 'No Churn': '#4682B4'},
                 title="Churn vs No Churn Probability", text='Probability')

    fig.update_layout(yaxis_range=[0, 1], template="plotly_white")
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
