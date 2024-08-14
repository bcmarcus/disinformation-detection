#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if service account file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_service_account_json>"
    exit 1
fi

SERVICE_ACCOUNT_FILE=$1

# Extract project ID from service account file
PROJECT_ID=$(jq -r '.project_id' "$SERVICE_ACCOUNT_FILE")

if [ -z "$PROJECT_ID" ]; then
    echo "Error: Could not extract project ID from service account file"
    exit 1
fi

echo "Using project ID: $PROJECT_ID"

# Authenticate with Google Cloud
gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_FILE"

# Set the project
gcloud config set project $PROJECT_ID

# Configure Docker to use gcloud as a credential helper
gcloud auth configure-docker -q

# Build and push the API server image
echo "Building and pushing API server Docker image..."
docker build -t gcr.io/$PROJECT_ID/python-server-api:latest -f run.dockerfile .
docker push gcr.io/$PROJECT_ID/python-server-api:latest

# Build and push the ML server image
echo "Building and pushing ML server Docker image..."
docker build -t gcr.io/$PROJECT_ID/ml-server:latest -f cuda.dockerfile .
docker push gcr.io/$PROJECT_ID/ml-server:latest

# Run the API server locally
echo "Running API server Docker container locally..."
docker run -d --name python-server-api-gcr \
  -p 8080:8080 \
  -e DEV=true \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/service_account.json \
  --network host \
  -v $(pwd)/$SERVICE_ACCOUNT_FILE:/app/service_account.json \
  gcr.io/$PROJECT_ID/python-server-api:latest

# Run the ML server locally
echo "Running ML server Docker container locally..."
docker run -d --name ml-server-gcr \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e DEV=true \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/service_account.json \
  --network host \
  -v $(pwd)/$SERVICE_ACCOUNT_FILE:/app/service_account.json \
  gcr.io/$PROJECT_ID/ml-server:latest

# Wait for the servers to start
echo "Waiting for servers to start..."
sleep 10

# Check if the API server container is running
if [ "$(docker ps -q -f name=python-server-api-gcr)" ]; then
    echo "API server is running at http://localhost:8080"
    echo "To stop the API server, run: docker stop python-server-api-gcr"
else
    echo "Error: API server Docker container failed to start. Printing logs..."
    docker logs python-server-api-gcr
fi

# Check if the ML server container is running
if [ "$(docker ps -q -f name=ml-server-gcr)" ]; then
    echo "ML server is running at http://localhost:5000"
    echo "To stop the ML server, run: docker stop ml-server-gcr"
else
    echo "Error: ML server Docker container failed to start. Printing logs..."
    docker logs ml-server-gcr
fi

echo "Script completed. Your production images are now running locally."