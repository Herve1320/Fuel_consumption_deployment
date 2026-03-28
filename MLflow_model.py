import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_compare():
    # 1. Setup Paths
    base_path = r"C:\Users\User\Downloads\Fuel_consumption_project"
    X_path = os.path.join(base_path, "data", "processed", "X_processed.csv")
    y_path = os.path.join(base_path, "data", "processed", "y.csv")
    
    # Path where all trained models will be stored locally
    models_dir = os.path.join(base_path, "models", "trained_models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("Error: Processed data not found. Run feature_engineering.py first!")
        return

    # 2. Load and Split Data
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Define the Models to Compare
    models = {
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    mlflow.set_experiment("Fuel_Consumption_Comparison")

    for model_name, model_obj in models.items():
        with mlflow.start_run(run_name=model_name):
            print(f"Training {model_name}...")
            
            # Train
            model_obj.fit(train_x, train_y)
            
            # Predict and Evaluate
            predictions = model_obj.predict(test_x)
            (rmse, mae, r2) = eval_metrics(test_y, predictions)

            # --- LOG TO MLFLOW ---
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model_obj, "model")
            else:
                mlflow.sklearn.log_model(model_obj, "model")

            # --- SAVE LOCALLY ---
            # This allows you to pick the best file manually from the folder
            if model_name == "XGBoost":
                model_save_path = os.path.join(models_dir, f"{model_name.lower()}_model.json")
                model_obj.save_model(model_save_path)
            else:
                model_save_path = os.path.join(models_dir, f"{model_name.lower()}_model.pkl")
                joblib.dump(model_obj, model_save_path)

            print(f"{model_name} Results -> RMSE: {rmse:.4f}, R2: {r2:.4f}")
            print(f"Model saved locally at: {model_save_path}")

    print(f"\nAll models trained and saved in: {models_dir}")
    print("Run 'mlflow ui --workers 1' to compare metrics.")

if __name__ == "__main__":
    train_and_compare()