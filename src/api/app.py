from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import uvicorn

app = FastAPI(title="Fuel Consumption API - demind.app")

# --- 1. SET DEFINITIVE PATHS ---
# Change from the hardcoded C:\ path to a relative path for Docker
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This will look for /app/models/production/...
MODEL_PATH = os.path.join("models", "production", "best_model.model")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")

# --- 2. INPUT DATA SCHEMA ---
class InputSchema(BaseModel):
    MODEL_YEAR: int
    ENGINE_SIZE: float
    CYLINDERS: int
    FUEL_TYPE: str
    TRANSMISSION: str
    VEHICLE_CLASS: str
    distance: float = 100.0
    load_weight: float = 1000.0

# --- 3. LOAD ARTIFACTS ON STARTUP ---
# This section runs once when the server starts
if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(f"CRITICAL: Preprocessor not found at {PREPROCESSOR_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"CRITICAL: Best model not found at {MODEL_PATH}. Did you run select_best_model.py?")

# Load the preprocessing pipeline (StandardScaler, OneHotEncoder)
preprocessor_pipeline = joblib.load(PREPROCESSOR_PATH)
# Extract the 'preprocessor' step from the saved Pipeline
preprocessor = preprocessor_pipeline.named_steps['preprocessor']

# Load the actual trained model
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

print("✅ Model and Preprocessor loaded successfully!")

# --- 4. PREDICTION ENDPOINT ---
@app.post("/predict")
def predict(data: InputSchema):
    # Convert Pydantic model to a Dictionary, then to a DataFrame
    # Using model_dump() for Pydantic v2 compatibility
    input_dict = data.model_dump()
    
    # Rename keys to match the CSV column names expected by your feature engineering
    # This maps 'MODEL_YEAR' to 'MODEL YEAR' etc.
    formatted_dict = {
        "MODEL YEAR": input_dict["MODEL_YEAR"],
        "ENGINE SIZE": input_dict["ENGINE_SIZE"],
        "CYLINDERS": input_dict["CYLINDERS"],
        "FUEL TYPE": input_dict["FUEL_TYPE"],
        "TRANSMISSION": input_dict["TRANSMISSION"],
        "VEHICLE CLASS": input_dict["VEHICLE_CLASS"],
        "distance": input_dict["distance"],
        "load_weight": input_dict["load_weight"]
    }
    
    df = pd.DataFrame([formatted_dict])
    
    # Feature Engineering logic
    df["vehicle_age"] = 2024 - df["MODEL YEAR"]
    
    # Transform using the preprocessor
    X_processed = preprocessor.transform(df)
    
    # Run Inference
    prediction = model.predict(X_processed)
    
    return {
        "status": "success",
        "prediction": float(prediction[0]),
        "unit": "L/100 km",
        "model_type": str(type(model).__name__)
    }

@app.get("/")
def home():
    return {"message": "Fuel Consumption API is online. Visit /docs for testing."}

if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)