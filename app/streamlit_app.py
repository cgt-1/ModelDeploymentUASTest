import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import streamlit as st
import pandas as pd

from src.inference import CreditScorePredictor


st.set_page_config(
    page_title="Credit Score Prediction",
    layout="centered"

)

st.title(
    "🏦 Credit Score Prediction System"
)

st.write(
    "Enter the customer's information below to predict their credit score."
)

predictor = CreditScorePredictor()


df = pd.read_csv("data/processed/cleaned_data.csv")

occupations = sorted(
    df["Occupation"]
    .dropna()
    .unique()
    .tolist()
)

credit_mix_options = sorted(
    df["Credit_Mix"]
    .dropna()
    .unique()
    .tolist()
)

payment_options = sorted(
    df["Payment_of_Min_Amount"]
    .dropna()
    .unique()
    .tolist()
)



st.header("👤 Personal Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

occupation = st.selectbox(
    "Occupation",
    occupations
)


st.header("💰 Income Information")

annual_income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

monthly_salary = st.number_input(
    "Monthly Inhand Salary",
    min_value=0.0,
    value=4000.0
)


st.header("🏦 Banking Information")

num_bank_accounts = st.number_input(
    "Number of Bank Accounts",
    min_value=0,
    value=3
)

num_credit_cards = st.number_input(
    "Number of Credit Cards",
    min_value=0,
    value=4
)

interest_rate = st.number_input(
    "Interest Rate",
    min_value=0.0,
    value=8.0
)

delay_due = st.number_input(
    "Delay from Due Date",
    min_value=0,
    value=10
)

changed_credit_limit = st.number_input(
    "Changed Credit Limit",
    value=5.0
)

credit_inquiries = st.number_input(
    "Number of Credit Inquiries",
    min_value=0,
    value=3
)

credit_mix = st.selectbox(
    "Credit Mix",
    credit_mix_options
)


st.header("📈 Credit Information")

outstanding_debt = st.number_input(
    "Outstanding Debt",
    min_value=0.0,
    value=500.0
)

credit_utilization = st.number_input(
    "Credit Utilization Ratio",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

credit_history = st.number_input(
    "Credit History Age (Months)",
    min_value=0,
    value=120
)

emi = st.number_input(
    "Total EMI per Month",
    min_value=0.0,
    value=150.0
)

monthly_balance = st.number_input(
    "Monthly Balance",
    value=500.0
)

loan_count = st.number_input(
    "Loan Count",
    min_value=0,
    value=2
)

payment_min = st.selectbox(
    "Payment of Minimum Amount",
    payment_options
)



user_input = {
    "Age": age,
    "Occupation": occupation,
    "Annual_Income": annual_income,
    "Monthly_Inhand_Salary": monthly_salary,
    "Num_Bank_Accounts": num_bank_accounts,
    "Num_Credit_Card": num_credit_cards,
    "Interest_Rate": interest_rate,
    "Delay_from_due_date": delay_due,
    "Changed_Credit_Limit": changed_credit_limit,
    "Num_Credit_Inquiries": credit_inquiries,
    "Credit_Mix": credit_mix,
    "Outstanding_Debt": outstanding_debt,
    "Credit_Utilization_Ratio": credit_utilization,
    "Credit_History_Age": credit_history,
    "Total_EMI_per_month": emi,
    "Monthly_Balance": monthly_balance,
    "Payment_of_Min_Amount": payment_min,
    "Loan_Count": loan_count
}


st.divider()

if st.button(
    "Predict Credit Score",
    use_container_width=True
):

    prediction = predictor.predict(user_input)

    st.subheader(
        "Prediction Result"
    )

    if prediction == "Good":

        st.success(
            f"✅ Predicted Credit Score: {prediction}"
        )

    elif prediction == "Standard":

        st.warning(
            f"🟡 Predicted Credit Score: {prediction}"
        )

    else:

        st.error(
            f"🔴 Predicted Credit Score: {prediction}"
        )