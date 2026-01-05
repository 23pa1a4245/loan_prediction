import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering

print("🚀 Training started")

# Load training data
data_path = os.path.join(PROJECT_ROOT, "data", "loan_sanction_train.csv")
df = pd.read_csv(data_path)

# Preprocess & feature engineering
df = preprocess_data(df)
df = feature_engineering(df, is_train=True)

# Split X and y
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

model.fit(X, y)

# Save model
model_path = os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl")
joblib.dump(model, model_path)

print("✅ Model trained and saved successfully")
