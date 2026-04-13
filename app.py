import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# load model and columns
model = joblib.load("model/model.pkl")
columns = joblib.load("model/columns.pkl")

# page setup
st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("Customer Churn Prediction Dashboard")
st.write("Enter customer details below:")

# input fields
age = st.slider("Age", 18, 80)
tenure = st.slider("Tenure", 0, 10)
usage = st.slider("Usage Frequency", 0, 100)
support_calls = st.slider("Support Calls", 0, 10)
payment_delay = st.slider("Payment Delay", 0, 50)
total_spend = st.number_input("Total Spend", 0.0)

gender = st.selectbox("Gender", ["Male", "Female"])
subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract = st.selectbox("Contract Length", ["Monthly", "Yearly"])

# prediction
if st.button("Predict"):

    input_data = {
        "Age": age,
        "Tenure": tenure,
        "Usage Frequency": usage,
        "Support Calls": support_calls,
        "Payment Delay": payment_delay,
        "Total Spend": total_spend,
        "Gender": gender,
        "Subscription Type": subscription,
        "Contract Length": contract
    }

    df = pd.DataFrame([input_data])

    # preprocessing
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    df = df.astype(float)

    # model prediction
    prediction = model.predict(df)[0]

    if prediction == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer will stay")

    # shap explainability
    st.subheader("Why this prediction?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    st.pyplot(fig)