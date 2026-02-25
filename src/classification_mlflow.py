import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,f1_score)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("EMI_Classification")

df = pd.read_csv("data/emi_prediction_dataset_cleaned.csv")

X = df.drop(['emi_eligibility', 'max_monthly_emi'], axis=1)
y = df['emi_eligibility']

le = LabelEncoder()
y = le.fit_transform(y)

cat_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

best_accuracy = 0
best_run_id = None
best_model_name = None
best_model_path = None
best_model = None


def evaluate_and_log(model, model_name, X_test, y_test):

    global best_accuracy, best_run_id, best_model_name, best_model_path, best_model

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, model_name)

    run_id = mlflow.active_run().info.run_id

    if acc > best_accuracy:
        best_accuracy = acc
        best_run_id = run_id
        best_model_name = model_name
        best_model_path = f"runs:/{run_id}/{model_name}"
        best_model = model

with mlflow.start_run(run_name="XGBoost_Classifier"):

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    evaluate_and_log(xgb_model, "xgb_model", X_test, y_test)

with mlflow.start_run(run_name="Logistic_Regression"):

    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train_scaled[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test_scaled[num_cols])

    log_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )

    log_model.fit(X_train_scaled, y_train)
    evaluate_and_log(log_model, "logistic_model", X_test_scaled, y_test)

# with mlflow.start_run(run_name="Random_Forest"):

#     rf_model = RandomForestClassifier(
#         n_estimators=200,
#         max_depth=None,
#         random_state=42,
#         n_jobs=-1,
#         class_weight="balanced"
#     )

#     rf_model.fit(X_train, y_train)
#     evaluate_and_log(rf_model, "random_forest_model", X_test, y_test)

if best_run_id is not None:
    result = mlflow.register_model(best_model_path, f"Best_Classification_Model_{best_model_name}")
    joblib.dump(best_model, "best_classification_model.pkl")

    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {best_accuracy}")

print("All classification models logged successfully!")