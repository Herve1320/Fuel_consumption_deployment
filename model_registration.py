# import mlflow
# from mlflow.entities import ViewType
# import joblib
# import shutil
# import os

import mlflow
from mlflow.entities import ViewType
import os

def export_and_register_champion():
    base_path = r"C:\Users\User\Downloads\Fuel_consumption_project"
    experiment_name = "Fuel_Consumption_Comparison"
    registered_model_name = "Fuel_Consumption_Model" # The name in the Registry
    alias_name = "Champion"

    # 1. Connect to MLflow and find the best run
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Experiment '{experiment_name}' not found!")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=["metrics.r2 DESC"] # Highest R2 wins
    )

    if not runs:
        print("No runs found in MLflow!")
        return

    best_run = runs[0]
    run_id = best_run.info.run_id
    r2_score = best_run.data.metrics.get('r2', 0)
    
    print(f"🏆 Found Best Run: {run_id} with R2: {r2_score:.4f}")

    # 2. Register the model in the MLflow Model Registry
    # This creates a "Version" (e.g., Version 1)
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri, registered_model_name)

    # 3. Assign the "Champion" Alias to this version
    # This allows your API to just ask for the 'Champion' without caring about the version number
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias_name,
        version=model_version.version
    )

    print(f"✅ Model '{registered_model_name}' Version {model_version.version} is now the '{alias_name}'!")

if __name__ == "__main__":
    export_and_register_champion()