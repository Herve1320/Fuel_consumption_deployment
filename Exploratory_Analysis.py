import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_analysis():
    # Define paths based on your project structure
    base_path = r"C:\Users\User\Downloads\Fuel_consumption_project"
    processed_data_path = os.path.join(base_path, "data", "processed", "X_processed.csv")
    target_data_path = os.path.join(base_path, "data", "processed", "y.csv")

    # Check if the files exist before loading
    if not os.path.exists(processed_data_path) or not os.path.exists(target_data_path):
        print(f"Error: Processed data files not found. Please run feature_engineering.py first.")
        return

    # Load the processed features and the target labels
    X = pd.read_csv(processed_data_path)
    y = pd.read_csv(target_data_path)

    # Combine them for visualization
    df_plot = X.copy()
    df_plot["FUEL_CONSUMPTION"] = y.values

    print("Data loaded successfully! Starting analysis...")
    print(df_plot.head())

    # Example Visualization: Distribution of Fuel Consumption
    plt.figure(figsize=(10, 6))
    sns.histplot(df_plot["FUEL_CONSUMPTION"], kde=True, color='blue')
    plt.title("Distribution of Fuel Consumption")
    plt.xlabel("Fuel Consumption (L/100 km)")
    plt.ylabel("Frequency")
    
    # Save the plot instead of just showing it (useful for remote environments)
    plt.savefig(os.path.join(base_path, "fuel_distribution.png"))
    plt.show()

    # Correlation Heatmap (only for numerical columns)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_plot.iloc[:, :10].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix (First 10 Features)")
    plt.savefig(os.path.join(base_path, "correlation_heatmap.png"))
    plt.show()

if __name__ == "__main__":
    run_analysis()