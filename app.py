import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="EMI Risk Assessment", layout="wide")
st.title("💳 EMI Prediction & Risk Assessment System")

classification_model = joblib.load("best_classification_model.pkl")
regression_model = joblib.load("best_regression_model.pkl")

df_train = pd.read_csv("data/emi_prediction_dataset_cleaned.csv")

X_train = df_train.drop(["emi_eligibility", "max_monthly_emi"], axis=1)
X_train = pd.get_dummies(X_train, drop_first=True)
model_columns = X_train.columns

st.header("📋 Enter Customer Financial Details")

reg_or_class = st.selectbox(
    "Select Model", ["Regression", "Classification"]
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    gender = st.selectbox("Gender", df_train["gender"].unique())
    marital_status = st.selectbox("Marital Status", df_train["marital_status"].unique())
    education = st.selectbox("Education", df_train["education"].unique())
    employment_type = st.selectbox("Employment Type", df_train["employment_type"].unique())
    company_type = st.selectbox("Company Type", df_train["company_type"].unique())
    house_type = st.selectbox("House Type", df_train["house_type"].unique())
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
    existing_loans = st.selectbox("Existing Loans", df_train["existing_loans"].unique())
    current_emi_amount = st.number_input("Current EMI Amount", 0, 100000, 5000)
    credit_score = st.number_input("Credit Score", 300, 850, 700)
    bank_balance = st.number_input("Bank Balance", 0, 1000000, 50000)
    emergency_fund = st.number_input("Emergency Fund", 0, 1000000, 20000)
    emi_scenario = st.selectbox("EMI Scenario", df_train["emi_scenario"].unique())

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
affordability_ratio = (monthly_salary - total_monthly_expenses) / monthly_salary if monthly_salary != 0 else 0

if st.button("Predict EMI Eligibility"):

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

    if reg_or_class == "Classification":
        eligibility = classification_model.predict(input_df)[0]
    else:
        max_emi = regression_model.predict(input_df)[0]

    st.success("Prediction Completed!")

    st.subheader("📊 Results")
    if reg_or_class == "Classification":
        name = ""
        if eligibility == 0:
            name = "Eligible"
        elif eligibility == 1:
            name = "High Risk"
        else:
            name = "Not Eligible"
        st.write(f"### EMI Eligibility: {name}")
    else:
        st.write(f"### Maximum Safe EMI: ₹ {round(max_emi, 2)}")