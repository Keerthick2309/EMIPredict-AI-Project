import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="EMI Risk Assessment", layout="wide")

classification_model = joblib.load("best_classification_model.pkl")
regression_model = joblib.load("best_regression_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.subheader("Enter Customer Financial Details")

reg_or_class = st.selectbox(
    "Select Prediction Type", ["Regression", "Classification"]
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("Personal Details")
    age = st.number_input("Age", value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    family_size = st.number_input("Family Size", value=4)
    dependents = st.number_input("Dependents", value=2)
    education = st.selectbox("Education", ["Graduate", "High School", "Post Graduate", "Professional"])

with col2:
    st.write("Employment & Housing")
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    company_type = st.selectbox("Company Type", ['MNC', 'Mid-size', 'Startup', 'Large Indian', 'Small'])
    years_of_employment = st.number_input("Years of Employment", value=10)
    house_type = st.selectbox("House Type",["Rented", "Own", "Family"])
    monthly_rent = st.number_input("Monthly Rent",value=10000)

with col3:
    st.write("Income & Expenses")
    monthly_salary = st.number_input("Monthly Salary",value=50000)
    school_fees = st.number_input("School Fees (Monthly)",value=0)
    college_fees = st.number_input("College Fees (Monthly)",value=0)
    travel_expenses = st.number_input( "Travel Expenses",value=3000)
    groceries_utilities = st.number_input("Groceries & Utilities",value=8000)
    other_monthly_expenses = st.number_input("Other Monthly Expenses",value=10000)

with col4:
    st.write("Loan & Credit Details")
    existing_loans = st.selectbox("Existing Loans",["No", "Yes"])
    current_emi_amount = st.number_input("Current EMI Amount",value=0)
    requested_amount = st.number_input("Requested Loan Amount",value=200000)
    requested_tenure = st.number_input("Requested Tenure (Months)", value=24)
    credit_score = st.number_input("Credit Score",value=800)
    bank_balance = st.number_input("Bank Balance", value=50000)
    emergency_fund = st.number_input("Emergency Fund",value=200000)
    emi_scenario = st.selectbox(
        "EMI Scenario",
        [
            "No",
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
    + current_emi_amount
)

debt_to_income_ratio = current_emi_amount / monthly_salary if monthly_salary != 0 else 0
expense_to_income_ratio = total_monthly_expenses / monthly_salary if monthly_salary != 0 else 0
affordability_ratio = ((monthly_salary - total_monthly_expenses) / monthly_salary if monthly_salary != 0 else 0)

if st.button("Predict", type="secondary"):
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

    st.session_state["input_dict"] = input_dict
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=False)

    st.success("Prediction Completed!")
    st.subheader("Results")

    if reg_or_class == "Classification":
        prediction = classification_model.predict(input_df)[0]
        st.session_state["prediction"] = prediction

        if prediction == 0:
            result = "Eligible"
        elif prediction == 1:
            result = "High Risk"
        else:
            result = "Not Eligible"

        st.write(f"### EMI Eligibility: {result}")

    else:
        max_emi = regression_model.predict(input_df)[0]
        st.session_state["prediction"] = max_emi
        st.write(f"### Maximum Safe EMI: ₹ {round(max_emi, 2)}")