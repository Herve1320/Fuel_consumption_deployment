# Step 1: Use an official lightweight Python image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies (needed for XGBoost/Scikit-learn)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your project files into the container
# We copy the src folder and the models folder
COPY src/ ./src/
COPY models/ ./models/

# Step 6: Expose the port FastAPI runs on
EXPOSE 8000

# Step 7: Command to run the API
# Note: We use 0.0.0.0 here so the container can talk to the outside world
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]