import mlflow
from mlflow.tracking import MlflowClient
import dagshub

# ----------------------------
# DAGSHUB / MLFLOW SETUP
# ----------------------------
dagshub.init(
    repo_owner='Herve1320',
    repo_name='Fuel_consumption',
    mlflow=True
)

# ----------------------------
# CONFIG
# ----------------------------
EXPERIMENT_NAME = "Fuel_Consumption_Comparison"
REGISTERED_MODEL_NAME = "Fuel_Consumption_Model"
ALIAS_NAME = "Champion"

def export_and_register_champion():
    client = MlflowClient()

    # 1. Get Experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found!")
        return

    # 2. Get Best Run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.r2 DESC"],
        max_results=1
    )

    if not runs:
        print("❌ No runs found.")
        return

    best_run = runs[0]
    run_id = best_run.info.run_id
    r2 = best_run.data.metrics.get("r2", 0)

    print(f"🏆 Best Run: {run_id} | R2: {r2:.4f}")

    # 3. Construct Source URI
    # This points to the "model" folder you defined in your training script
    model_uri = f"runs:/{run_id}/model"

    try:
        # 4. Check if the Registered Model exists, if not, create it
        try:
            client.get_registered_model(REGISTERED_MODEL_NAME)
        except:
            print(f"Creating registered model '{REGISTERED_MODEL_NAME}'...")
            client.create_registered_model(REGISTERED_MODEL_NAME)

        # 5. Create the Model Version
        # Using client.create_model_version is more robust than mlflow.register_model
        print(f"Registering version for {run_id}...")
        model_version = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=model_uri,
            run_id=run_id
        )

        # 6. Set Alias
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias=ALIAS_NAME,
            version=model_version.version
        )

        print(f"✅ Successfully registered version {model_version.version}")
        print(f"✅ Alias '{ALIAS_NAME}' assigned to version {model_version.version}")

    except Exception as e:
        print(f"❌ Error during registration: {e}")

if __name__ == "__main__":
    export_and_register_champion()