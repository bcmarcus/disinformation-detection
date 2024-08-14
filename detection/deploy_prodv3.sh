#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

set -x

# Check if the necessary arguments are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_service_account_json> [--recreate-ml] [--recreate-run]"
    exit 1
fi

SERVICE_ACCOUNT_FILE=$1
RECREATE_ML_FLAG=""
RECREATE_RUN_FLAG=""

# Parse flags
for arg in "$@"
do
    case $arg in
        --recreate-ml)
        RECREATE_ML_FLAG="--recreate-ml"
        shift
        ;;
        --recreate-run)
        RECREATE_RUN_FLAG="--recreate-run"
        shift
        ;;
    esac
done

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

# Set your Google Compute Engine instance details
GCE_INSTANCE_NAME="ml-instance"
GCE_ZONE="asia-southeast1-c"

# Authenticate with Google Cloud
gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_FILE"

# Set the project
gcloud config set project $PROJECT_ID

# Build the Docker image for the ML server using cuda.dockerfile
echo "Building ML server Docker image..."
docker build -t gcr.io/$PROJECT_ID/ml-server:latest -f cuda.dockerfile .

# Push the ML server image to Google Container Registry
echo "Pushing ML server image to Google Container Registry..."
gcloud auth configure-docker -q
docker push gcr.io/$PROJECT_ID/ml-server:latest

# Check if the GCE instance already exists
INSTANCE_EXISTS=$(gcloud compute instances list --filter="name=($GCE_INSTANCE_NAME)" --zones=$GCE_ZONE --format="value(name)")

ML_INSTANCE_STATUS=""

if [ -n "$INSTANCE_EXISTS" ] && [ "$RECREATE_ML_FLAG" == "--recreate-ml" ]; then
    echo "GCE instance $GCE_INSTANCE_NAME exists and --recreate-ml flag is set. Deleting instance..."
    gcloud compute instances delete $GCE_INSTANCE_NAME --zone=$GCE_ZONE --quiet
    INSTANCE_EXISTS=""
    ML_INSTANCE_STATUS="recreated"
fi


if [ -z "$INSTANCE_EXISTS" ]; then
    # Deploy ML server to Google Compute Engine
    echo "Deploying ML server to Google Compute Engine..."
    gcloud compute instances create $GCE_INSTANCE_NAME \
    --zone=$GCE_ZONE \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --maintenance-policy=TERMINATE \
    --preemptible \
    --boot-disk-size=200GB \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --metadata="install-nvidia-driver=True" \
    --metadata-from-file startup-script=gpu_startup_script.sh

    echo "ML server instance created successfully: $GCE_INSTANCE_NAME"
    if [ "$ML_INSTANCE_STATUS" != "recreated" ]; then
        ML_INSTANCE_STATUS="created"
    fi

    # Wait for the instance to be ready
    echo "Waiting for instance to be ready..."
    while true; do
        STATUS=$(gcloud compute instances describe $GCE_INSTANCE_NAME --zone=$GCE_ZONE --format="value(status)")
        if [ "$STATUS" = "RUNNING" ]; then
            echo "Instance is running."
            break
        fi
        echo "Instance status: $STATUS. Waiting..."
        sleep 10
    done

    # Wait for the startup script to complete
    echo "Waiting for startup script to complete..."
    while true; do
        if gcloud compute ssh $GCE_INSTANCE_NAME --zone=$GCE_ZONE --command="ls /var/log/startup-script-complete" &> /dev/null; then
            echo "Startup script completed."
            break
        fi
        echo "Startup script still running. Waiting..."
        sleep 30
    done

    # Copy the Docker image to the instance
    echo "Copying Docker image to instance..."
    gcloud compute scp cuda.dockerfile $GCE_INSTANCE_NAME:/tmp/cuda.dockerfile --zone=$GCE_ZONE
    gcloud compute scp env.yml $GCE_INSTANCE_NAME:/tmp/env.yml --zone=$GCE_ZONE
    gcloud compute scp server/cuda.py $GCE_INSTANCE_NAME:/tmp/server/cuda.py --zone=$GCE_ZONE

    echo "ML server deployed and container started on GCE instance: $GCE_INSTANCE_NAME"
else
    echo "GCE instance $GCE_INSTANCE_NAME already exists. Skipping creation."
    ML_INSTANCE_STATUS="already existed"
fi

# Function to get the external IP of the GCE instance
get_gce_external_ip() {
    gcloud compute instances describe $GCE_INSTANCE_NAME \
        --zone=$GCE_ZONE \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
}

# After ML server deployment, get its IP
GCE_IP=$(get_gce_external_ip)

if [ -z "$GCE_IP" ]; then
    echo "Error: Could not retrieve external IP for GCE instance"
    exit 1
fi

# Construct LLM handler URL
LLM_HANDLER_URL="http://${GCE_IP}:5000"

echo "LLM Handler URL: $LLM_HANDLER_URL"

# Build the Docker image for the API server using run.dockerfile
echo "Building API server Docker image..."
docker build -t gcr.io/$PROJECT_ID/python-server-api:latest -f run.dockerfile .

# Push the image to Google Container Registry
echo "Pushing API server image to Google Container Registry..."
docker push gcr.io/$PROJECT_ID/python-server-api:latest

# Deploy to Google Cloud Run if flag is set
if [ "$RECREATE_RUN_FLAG" == "--recreate-run" ]; then
    echo "Deploying to Google Cloud Run..."
    gcloud run deploy python-server-api-gcr \
      --image gcr.io/$PROJECT_ID/python-server-api:latest \
      --platform managed \
      --region asia-southeast1 \
      --set-env-vars LLM_HANDLER_URL=$LLM_HANDLER_URL \
      --allow-unauthenticated \
      --port=8080 \
      --cpu=2 \
      --memory=4Gi \
      --timeout=3600 \
      --concurrency=80

    # Get the URL of the deployed service
    SERVICE_URL=$(gcloud run services describe python-server-api-gcr --platform managed --region asia-southeast1 --format 'value(status.url)')

    echo "\n\n"

    echo "API server deployed successfully. URL: $SERVICE_URL"

    echo "\n--- Deployment Summary ---"
    echo "ML Instance: $ML_INSTANCE_STATUS"
    echo "Google Cloud Run: Deployed"
    echo "API Server URL: $SERVICE_URL"
    echo "-------------------------"
else
    echo "Skipping Google Cloud Run deployment."
fi

echo "Production deployment complete."
