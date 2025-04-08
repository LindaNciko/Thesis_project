FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including Git and Git LFS
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p /app/model

# Copy model files individually with verification
COPY model/feature_selector.joblib /app/model/
COPY model/multi_output_model.joblib /app/model/
COPY model/label_encoders.joblib /app/model/
COPY model/inverse_maps.joblib /app/model/
COPY model/feature_cols.joblib /app/model/

# Verify model files exist
RUN ls -lh /app/model/

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"] 