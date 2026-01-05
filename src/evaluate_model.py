import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering

print("🔍 Evaluation started")

# -------------------------------------------------
# Load test dataset
# -------------------------------------------------
test_path = os.path.join(PROJECT_ROOT, "data", "loan_sanction_test.csv")
df_test = pd.read_csv(test_path)

print(f"Test data shape: {df_test.shape}")

# -------------------------------------------------
# Preprocessing & feature engineering
# -------------------------------------------------
df_test = preprocess_data(df_test)
df_test = feature_engineering(df_test, is_train=False)

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model_path = os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl")
model = joblib.load(model_path)

print("✅ Model loaded")

# -------------------------------------------------
# CASE 1: Loan_Status EXISTS (true evaluation)
# -------------------------------------------------
if "Loan_Status" in df_test.columns:
    X_test = df_test.drop("Loan_Status", axis=1)
    y_test = df_test["Loan_Status"]

    y_pred = model.predict(X_test)

    print("\n📊 Model Evaluation Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------------------------
# CASE 2: Loan_Status DOES NOT EXIST (prediction only)
# -------------------------------------------------
else:
    predictions = model.predict(df_test)
    df_test["Predicted_Loan_Status"] = predictions

    # Map 0/1 to N/Y
    df_test["Predicted_Loan_Status"] = df_test["Predicted_Loan_Status"].map(
        {1: "Approved", 0: "Rejected"}
    )

    print("\n⚠️ Loan_Status not found in test data")
    print("Showing sample predictions:\n")
    print(df_test[["Predicted_Loan_Status"]].head())

    # Save predictions
    output_path = os.path.join(PROJECT_ROOT, "data", "loan_predictions.csv")
    df_test.to_csv(output_path, index=False)

    print(f"\n📁 Predictions saved to: {output_path}")

print("✅ Evaluation completed")
