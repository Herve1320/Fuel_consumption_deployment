import pandas as pd

# 1. Load the dataset
df = pd.read_csv('fuel_consumption2015-2024.csv')

# 2. Handle Duplicates
# Removes identical rows to ensure data quality
df = df.drop_duplicates()

# 3. Convert Rating Columns to Numeric
# We use errors='coerce' to turn any 'n/a' strings into NaN (nulls)
df['CO2 rating'] = pd.to_numeric(df['CO2 rating'], errors='coerce')
df['Smog rating'] = pd.to_numeric(df['Smog rating'], errors='coerce')

# 4. Handle Missing Values using Median
# This calculates the middle value of existing ratings and fills the gaps
co2_median = df['CO2 rating'].median()
smog_median = df['Smog rating'].median()

df['CO2 rating'] = df['CO2 rating'].fillna(co2_median)
df['Smog rating'] = df['Smog rating'].fillna(smog_median)

# 5. Save the cleaned dataset
# Column names are preserved exactly as they were in the original file
output_filename = 'cleaned_fuel_consumption_median.csv'
df.to_csv(output_filename, index=False)

print(f"Cleaning complete. Missing values filled with Medians (CO2: {co2_median}, Smog: {smog_median}).")
print(f"File saved as: {output_filename}")