import mlflow
from mlflow.tracking import MlflowClient
import os

# --- CONFIGURATION ---
# Points to the MLflow server running on your host machine
MLFLOW_TRACKING_URI = "http://host.docker.internal:5000"
EXPERIMENT_NAME = "Fuel_Consumption_Comparison"
REGISTERED_MODEL_NAME = "Fuel_Consumption_Model" 
ALIAS_NAME = "Champion"

def export_and_register_champion():
    """
    Finds the best performing model in MLflow and labels it as 'Champion'
    for the Production API to use.
    """
    # 1. Connect to the MLflow Network Server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # 2. Find the experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found! Ensure training has run.")
        return

    # 3. Search for the best run based on R2 Score
    # We order by R2 DESC so the highest value is the first result
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=["metrics.r2 DESC"]
    )

    if not runs:
        print("⚠️ No runs found in MLflow. Nothing to register.")
        return

    best_run = runs[0]
    run_id = best_run.info.run_id
    r2_score = best_run.data.metrics.get('r2', 0)
    
    print(f"🏆 Best Run Found: {run_id} | R2 Score: {r2_score:.4f}")

    # 4. Register this specific version in the Model Registry
    # This creates or updates 'Fuel_Consumption_Model' in the MLflow UI
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)

    # 5. Assign the 'Champion' Alias
    # Your FastAPI will now pull this specific version automatically
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=ALIAS_NAME,
        version=model_version.version
    )

    print(f"✅ Version {model_version.version} successfully promoted to '{ALIAS_NAME}'!")

if __name__ == "__main__":
    export_and_register_champion()