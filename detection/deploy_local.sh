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

# Check if the user is logged into Docker
if ! docker info > /dev/null 2>&1; then
    echo "Docker login required. Please enter your Docker credentials."
    docker login
    if [ $? -ne 0 ]; then
        echo "Docker login failed."
        exit 1
    fi
else
    echo "Already logged into Docker."
fi

# Build the Docker image using run.dockerfile
echo "Building API server Docker image..."
docker build -t python-server-api:dev -f run.dockerfile .

# Check if the container already exists and remove it if it does
if [ "$(docker ps -aq -f name=python-server-api-dev)" ]; then
    echo "Stopping and removing existing Docker container..."
    docker stop python-server-api-dev
    docker rm python-server-api-dev
fi

# Run the Docker container
echo "Running API server Docker container..."
docker run -d --name python-server-api-dev \
  -p 8080:8080 \
  -e DEV=true \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/service_account.json \
  --network host \
  -v $(pwd)/$SERVICE_ACCOUNT_FILE:/app/service_account.json \
  python-server-api:dev

# Build the Docker image for the ML server using cuda.dockerfile
echo "Building ML server Docker image..."
docker build -t ml-server:dev -f cuda.dockerfile .

# Check if the container already exists and remove it if it does
if [ "$(docker ps -aq -f name=ml-server-dev)" ]; then
    echo "Stopping and removing existing Docker container..."
    docker stop ml-server-dev
    docker rm ml-server-dev
fi

# Run the Docker container
echo "Running ML server Docker container..."
docker run -d --name ml-server-dev \
  --gpus all \
  -e DEV=true \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/service_account.json \
  --network host \
  -v $(pwd)/$SERVICE_ACCOUNT_FILE:/app/service_account.json \
  ml-server:dev

# Wait for the servers to start
echo "Waiting for servers to start..."
sleep 10

# Check if the API server container is running
if [ "$(docker ps -q -f name=python-server-api-dev)" ]; then
    # Run the test script
    echo "Running test script on API server..."
    docker exec python-server-api-dev micromamba run -n dis2 python /app/server/test_server.py

    echo "Local deployment complete. API server is running at http://localhost:8080"
    echo "To stop the API server, run: docker stop python-server-api-dev"
else
    echo "Error: API server Docker container failed to start. Printing logs..."
    docker logs python-server-api-dev
    exit 1
fi

# Check if the ML server container is running
if [ "$(docker ps -q -f name=ml-server-dev)" ]; then
    echo "Local deployment complete. ML server is running at http://localhost:5000"
    echo "To stop the ML server, run: docker stop ml-server-dev"
else
    echo "Error: ML server Docker container failed to start. Printing logs..."
    docker logs ml-server-dev
    exit 1
fi
