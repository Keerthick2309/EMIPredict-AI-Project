import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="EMI Risk Assessment", layout="wide")
st.title("EMI Prediction & Risk Assessment System")

classification_model = joblib.load("best_classification_model.pkl")
regression_model = joblib.load("best_regression_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.header("Enter Customer Financial Details")

reg_or_class = st.selectbox(
    "Select Prediction Type", ["Classification", "Regression"]
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    company_type = st.selectbox("Company Type", ["Small", "Medium", "Large"])
    house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
    requested_tenure = st.number_input("Requested Tenure (Months)", 3, 120, 24)

with col2:
    monthly_salary = st.number_input("Monthly Salary", 10000, 500000, 50000)
    years_of_employment = st.number_input("Years of Employment", 0, 40, 5)
    monthly_rent = st.number_input("Monthly Rent", 0, 200000, 10000)
    family_size = st.number_input("Family Size", 1, 15, 3)
    dependents = st.number_input("Dependents", 0, 10, 1)
    school_fees = st.number_input("School Fees", 0, 100000, 2000)
    college_fees = st.number_input("College Fees", 0, 100000, 3000)
    requested_amount = st.number_input("Requested Loan Amount", 10000, 2000000, 200000)

with col3:
    travel_expenses = st.number_input("Travel Expenses", 0, 100000, 3000)
    groceries_utilities = st.number_input("Groceries & Utilities", 0, 100000, 8000)
    other_monthly_expenses = st.number_input("Other Monthly Expenses", 0, 100000, 2000)
    existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
    current_emi_amount = st.number_input("Current EMI Amount", 0, 100000, 5000)
    credit_score = st.number_input("Credit Score", 300, 850, 700)
    bank_balance = st.number_input("Bank Balance", 0, 1000000, 50000)
    emergency_fund = st.number_input("Emergency Fund", 0, 1000000, 20000)
    emi_scenario = st.selectbox(
        "EMI Scenario",
        [
            "E-commerce Shopping EMI",
            "Home Appliances EMI",
            "Vehicle EMI",
            "Personal Loan EMI",
            "Education EMI",
        ],
    )

total_monthly_expenses = (
    monthly_rent
    + school_fees
    + college_fees
    + travel_expenses
    + groceries_utilities
    + other_monthly_expenses
)

debt_to_income_ratio = current_emi_amount / monthly_salary if monthly_salary != 0 else 0
expense_to_income_ratio = total_monthly_expenses / monthly_salary if monthly_salary != 0 else 0
affordability_ratio = (
    (monthly_salary - total_monthly_expenses) / monthly_salary
    if monthly_salary != 0
    else 0
)

if st.button("Predict"):

    input_dict = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "total_monthly_expenses": total_monthly_expenses,
        "debt_to_income_ratio": debt_to_income_ratio,
        "expense_to_income_ratio": expense_to_income_ratio,
        "affordability_ratio": affordability_ratio,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    st.success("Prediction Completed!")

    st.subheader("Results")

    if reg_or_class == "Classification":
        prediction = classification_model.predict(input_df)[0]

        if prediction == 0:
            result = "Eligible"
        elif prediction == 1:
            result = "High Risk"
        else:
            result = "Not Eligible"

        st.write(f"### EMI Eligibility: {result}")

    else:
        max_emi = regression_model.predict(input_df)[0]
        st.write(f"### Maximum Safe EMI: ₹ {round(max_emi, 2)}")