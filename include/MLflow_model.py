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
import os

# --- CONFIGURATION ---
# Use the internal Docker network name 'postgres' to connect to the DB
DB_URL = "postgresql://airflow:airflow@postgres:5432/airflow"
# Point to your MLflow container or host
MLFLOW_TRACKING_URI = "http://host.docker.internal:5000"

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_compare():
    # 1. Setup MLflow Tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Fuel_Consumption_Comparison")

    # 2. Load Data from PostgreSQL
    print("🐘 Connecting to PostgreSQL to fetch training data...")
    try:
        engine = create_engine(DB_URL)
        # Assuming your processed data is stored in a table called 'fuel_processed'
        df = pd.read_sql("SELECT * FROM fuel_processed", engine) 
    except Exception as e:
        print(f"❌ Database Connection Error: {e}")
        return

    # 3. Prepare Data
    # Assuming 'target' is the name of your fuel consumption column
    X = df.drop(columns=['target'])
    y = df['target'].values.ravel()
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Define Models
    models = {
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    for model_name, model_obj in models.items():
        with mlflow.start_run(run_name=model_name):
            print(f"Training {model_name}...")
            
            # Train
            model_obj.fit(train_x, train_y)
            
            # Predict and Evaluate
            predictions = model_obj.predict(test_x)
            (rmse, mae, r2) = eval_metrics(test_y, predictions)

            # --- LOG TO MLFLOW SERVER VIA HTTP ---
            mlflow.log_params(model_obj.get_params() if hasattr(model_obj, 'get_params') else {})
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            
            # Log Model to the HTTP Registry
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model_obj, "model")
            else:
                mlflow.sklearn.log_model(model_obj, "model")

            print(f"✅ {model_name} logged to MLflow (R2: {r2:.4f})")

    print("\n🚀 All models trained. Check your dashboard at http://localhost:5000")

if __name__ == "__main__":
    train_and_compare()