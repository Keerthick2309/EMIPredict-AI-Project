import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("EMI_Regression")

df = pd.read_csv("data/emi_prediction_dataset_cleaned.csv")

best_rmse = float("inf")
best_run_id = None
best_model_name = None
best_model_path = None
best_model = None

def evaluate_and_log(model, model_name, X_test, y_test):
    global best_rmse, best_model_name, best_model_path, best_run_id, best_model

    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_params(model.get_params())
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, model_name)

    run_id = mlflow.active_run().info.run_id

    if rmse < best_rmse:
        best_rmse = rmse
        best_run_id = run_id
        best_model_name = model_name
        best_model_path = f"runs:/{run_id}/{model_name}"
        best_model = model


with mlflow.start_run(run_name="Linear_Regression"):

    X_lin = df.drop([
        "emi_eligibility",
        "max_monthly_emi",
        "school_fees",
        "college_fees",
        "travel_expenses",
        "groceries_utilities",
        "other_monthly_expenses",
        "monthly_rent",
        "current_emi_amount"
    ], axis=1)

    y_lin = df["max_monthly_emi"]

    X_lin = pd.get_dummies(X_lin, drop_first=True)

    X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
        X_lin, y_lin, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_lin)
    X_test_scaled = scaler.transform(X_test_lin)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train_lin)

    evaluate_and_log(lin_reg, "linear_model", X_test_scaled, y_test_lin)

with mlflow.start_run(run_name="XGBoost_Regressor"):

    X = df.drop(["emi_eligibility", "max_monthly_emi"], axis=1)
    y = df["max_monthly_emi"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb_model = XGBRegressor(   
        n_estimators=800,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    evaluate_and_log(xgb_model, "xgb_model", X_test, y_test)

with mlflow.start_run(run_name="Random_Forest_Regressor"):

    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs= -1
    )

    rf_model.fit(X_train, y_train)

    evaluate_and_log(rf_model, "rf_model", X_test, y_test)


if best_run_id is not None:
    mlflow.register_model(best_model_path, f"Best_Regression_Model_{best_model_name}")
    joblib.dump(best_model, "best_regression_model.pkl")
    
    print("Best Model Selected:", best_model_name)
    print("Best RMSE:", round(best_rmse, 2))

print("All regression models logged successfully!")