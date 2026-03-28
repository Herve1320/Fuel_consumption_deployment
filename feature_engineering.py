import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def load_data(path):
    """Loads CSV and standardizes column names to handle case-sensitivity and extra spaces."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"The dataset was not found at: {path}")
        
    df = pd.read_csv(path)
    
    # Standardize column names: Strip whitespace and convert to UPPERCASE
    df.columns = df.columns.str.strip().str.upper()
    
    # Map specific names from your dataset to the variable names used in the script
    rename_map = {
        "ENGINE SIZE (L)": "ENGINE SIZE",
        "COMBINED (L/100 KM)": "FUEL CONSUMPTION"
    }
    df = df.rename(columns=rename_map)
    return df

def create_features(df):
    """Calculates vehicle age and ensures required synthetic features exist."""
    # Now df["MODEL YEAR"] works because of the normalization in load_data
    df["vehicle_age"] = 2024 - df["MODEL YEAR"]

    # Ensure synthetic columns exist if not provided in the source data
    if "distance" not in df.columns:
        df["distance"] = 100

    if "load_weight" not in df.columns:
        df["load_weight"] = 1000

    return df

def build_preprocessor(df):
    """Defines the scaling and encoding logic for the pipeline."""
    categorical_cols = ["FUEL TYPE", "TRANSMISSION", "VEHICLE CLASS"]
    numerical_cols = [
        "ENGINE SIZE",
        "CYLINDERS",
        "vehicle_age",
        "distance",
        "load_weight",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocessor

def run_feature_engineering(input_path, output_path, pipeline_path):
    """Main execution flow for processing data and saving artifacts."""
    # 1. Load and Clean
    df = load_data(input_path)
    df = create_features(df)

    # 2. Split Features and Target
    if "FUEL CONSUMPTION" not in df.columns:
        available = df.columns.tolist()
        raise KeyError(f"Target 'FUEL CONSUMPTION' not found. Available columns: {available}")

    X = df.drop(columns=["FUEL CONSUMPTION"])
    y = df["FUEL CONSUMPTION"]

    # 3. Fit Pipeline
    preprocessor = build_preprocessor(df)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_processed = pipeline.fit_transform(X)

    # 4. Save Processed Data (CSV)
    os.makedirs(output_path, exist_ok=True)
    processed_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed)
    processed_df.to_csv(os.path.join(output_path, "X_processed.csv"), index=False)
    y.to_csv(os.path.join(output_path, "y.csv"), index=False)

    # 5. Save Pipeline (PKL) - Ensures the 'models' directory exists
    model_dir = os.path.dirname(pipeline_path)
    if model_dir: # Only create if there's a directory component in the path
        os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(pipeline, pipeline_path)

    print(f"Success! Processed data saved to: {output_path}")
    print(f"Pipeline saved to: {pipeline_path}")

if __name__ == "__main__":
    # Updated to use the paths consistent with your local environment
    BASE_DIR = r"C:\\Users\\User\\Downloads\\Fuel_consumption_project"
    
    run_feature_engineering(
        input_path=os.path.join(BASE_DIR, "fuel_consumption2015-2024.csv"),
        output_path=os.path.join(BASE_DIR, "data", "processed"),
        pipeline_path=os.path.join(BASE_DIR, "models", "preprocessor.pkl"),
    )


# if __name__ == "__main__":
#     run_feature_engineering(
#         "C:\\Users\\User\\Downloads\\Fuel_consumption_project\\fuel_consumption2015-2024.csv",
#         "C:\\Users\\User\\Downloads\\Fuel_consumption_project\\data\\processed",
#         "C:\\Users\\User\\Downloads\\Fuel_consumption_project\\models\\preprocessor.pkl",
#     )