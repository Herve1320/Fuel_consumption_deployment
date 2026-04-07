import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import dagshub
import os

# --- DAGSHUB SETUP ---
dagshub.init(
    repo_owner='Herve1320',
    repo_name='Fuel_consumption',
    mlflow=True
)

# --- CONFIGURATION ---
DB_URL = "postgresql://admin:admin@127.0.0.1:5435/admin"

# ⚠️ REMOVE local MLflow URI → DagsHub handles it automatically
# mlflow.set_tracking_uri(...) NOT NEEDED

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_compare():

    # Set experiment (this will be created in DagsHub)
    mlflow.set_experiment("Fuel_Consumption_Comparison")

    # --- LOAD DATA ---
    print("🐘 Connecting to PostgreSQL...")

    try:
        engine = create_engine(DB_URL)
        df = pd.read_sql("SELECT * FROM fuel_processed", engine)
    except Exception as e:
        print(f"❌ Database Connection Error: {e}")
        return
    print("Columns in dataset:")
    print(df.columns.tolist())
    # --- PREPARE DATA ---
    target_col = "CO2 emissions (g/km)"

    if target_col not in df.columns:
        print(f"❌ Target column '{target_col}' not found in dataset")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X = pd.get_dummies(X)


    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- MODELS ---
    models = {
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            print(f"🚀 Training {model_name}...")

            model.fit(train_x, train_y)

            predictions = model.predict(test_x)
            rmse, mae, r2 = eval_metrics(test_y, predictions)

            # --- LOGGING ---
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # --- LOG MODEL ---
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"✅ {model_name} → R2: {r2:.4f}")

    print("\n🎯 All models logged to DagsHub MLflow!")

if __name__ == "__main__":
    train_and_compare()