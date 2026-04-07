import pandas as pd
from sqlalchemy import create_engine

DB_URL = "postgresql://admin:admin@127.0.0.1:5435/admin"
engine = create_engine(DB_URL)

# 🔁 Replace with your actual dataset file
df = pd.read_csv(r"C:\Users\User\Downloads\Fuel_consumption_project\cleaned_fuel_consumption_median.csv")

# Optional: ensure your target column exists
print(df.columns)

# Load into PostgreSQL
df.to_sql("fuel_processed", engine, if_exists="replace", index=False)

print("✅ Table fuel_processed created and data loaded")