import streamlit as st
import matplotlib.pyplot as plt
import joblib
from xgboost import plot_importance
import pandas as pd
import seaborn as sns

st.title("Analytics Dashboard")

regression_model = joblib.load("best_regression_model.pkl")
classification_model = joblib.load("best_classification_model.pkl")

#df = pd.read_csv("data/emi_prediction_dataset_cleaned.csv")

# col10, col11 = st.columns(2)

# with col10:
#     st.subheader("EMI Eligibility Distribution")
#     fig, ax = plt.subplots()
#     sns.countplot(x='emi_eligibility', data=df, ax=ax)
#     plt.title("EMI Eligibility Distribution")
#     plt.xlabel("EMI Eligibility")
#     plt.ylabel("Count")
#     st.pyplot(fig)

# with col11:
#     st.subheader("Max Monthly EMI Distribution")
#     fig10, ax10 = plt.subplots()
#     sns.histplot(df['max_monthly_emi'], bins=50, ax=ax10)
#     plt.title("Max Monthly EMI Distribution")
#     st.pyplot(fig10)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Regression Feature Importance")
    fig1, ax1 = plt.subplots()
    plot_importance(
        regression_model,
        ax=ax1,
        max_num_features=20
    )
    ax1.set_title("Top 20 Important Features")
    st.pyplot(fig1)

with col2:
    st.subheader("Classification Feature Importance")
    fig2, ax2 = plt.subplots()
    plot_importance(
        classification_model,
        ax=ax2,
        max_num_features=20
    )
    ax2.set_title("Top 20 Important Features")
    st.pyplot(fig2)

if "input_dict" not in st.session_state:
    pass
else:
    col1, col2 = st.columns(2)

    data = st.session_state["input_dict"]
    salary = data["monthly_salary"]
    expenses = data["total_monthly_expenses"]
    prediction = st.session_state.get("prediction", 0)

    with col1:
        st.subheader("Income vs Expenses")
        fig1, ax1 = plt.subplots()
        ax1.bar(["Salary", "Expenses"], [salary, expenses])
        ax1.set_title("Income vs Expenses")
        st.pyplot(fig1)

    with col2:
        st.subheader("EMI Comparison")
        fig2, ax2 = plt.subplots()
        ax2.bar(
            ["Current EMI", "Eligible EMI"],
            [data["current_emi_amount"], prediction]
        )
        ax2.set_title("EMI Comparison")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Expense Breakdown")
        expenses = {
            "Rent": data["monthly_rent"],
            "School Fees": data["school_fees"],
            "College Fees": data["college_fees"],
            "Travel": data["travel_expenses"],
            "Groceries": data["groceries_utilities"],
            "Other": data["other_monthly_expenses"],
            "Current EMI": data["current_emi_amount"]
        }

        fig, ax = plt.subplots()
        ax.pie(expenses.values(), labels=expenses.keys(), autopct="%1.1f%%")
        ax.set_title("Expense Breakdown")
        st.pyplot(fig)
    
    with col4:
        st.subheader("Income Analysis")
        disposable_income = data["monthly_salary"] - data["total_monthly_expenses"]
        labels = ["Salary", "Expenses", "Disposable"]
        values = [
            data["monthly_salary"],
            data["total_monthly_expenses"],
            disposable_income
        ]
        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_title("Income Analysis")
        st.pyplot(fig)