from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# 1. Path Setup: Astro CLI maps your project root to /usr/local/airflow
# We add 'include' so Airflow can find your ML scripts
sys.path.insert(0, '/usr/local/airflow/include')

# 2. Corrected Imports based on your uploaded files
from MLflow_model import train_and_compare
from model_registration import export_and_register_champion

default_args = {
    'owner': 'demind_team',
    'start_date': datetime(2024, 3, 30),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def train_task():
    """
    Triggers the MLflow training for ElasticNet, RF, and XGBoost.
    """
    print("🚀 Starting MLflow Training Comparison...")
    # Matches the function name in your MLflow_model.py
    train_and_compare() 

def select_best():
    """
    Compares R2 scores and updates the 'Champion' alias in MLflow.
    """
    print("🏆 Finding the best model and registering as Champion...")
    # Matches the function name in your model_registration.py
    export_and_register_champion()

with DAG(
    'fuel_ml_automation', 
    default_args=default_args, 
    schedule_interval='@weekly',
    catchup=False
) as dag:
    
    t1 = PythonOperator(
        task_id='train_models', 
        python_callable=train_task
    )
    
    t2 = PythonOperator(
        task_id='promote_champion', 
        python_callable=select_best
    )

    t1 >> t2  # Set the workflow sequence