#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if service account file and service name are provided
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

# Function to check if a command was successful
check_success() {
    if [ $? -eq 0 ]; then
        echo "Success: $1"
    else
        echo "Error: $1 failed"
    fi
}

# Function to check if a Cloud Run service exists
service_exists() {
    if gcloud run services describe python-server-api-gcr --platform managed --region asia-southeast1 &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to list and stop all Google Compute Engine instances with the given name
stop_all_instances() {
    local instance_name=$1
    local instances=$(gcloud compute instances list --filter="name:$instance_name" --format="value(name,zone)")

    if [ -z "$instances" ]; then
        echo "No instances found with name $instance_name."
        return
    fi

    echo "Stopping all instances with name $instance_name..."
    while read -r instance zone; do
        gcloud compute instances delete "$instance" --zone="$zone" --quiet
        check_success "Compute Engine instance $instance in zone $zone shutdown"
    done <<< "$instances"
}

# Shutdown Google Cloud Run service
echo "Shutting down Cloud Run service..."
if service_exists python-server-api-gcr; then
    gcloud run services delete python-server-api-gcr --platform managed --region asia-southeast1 --quiet
    check_success "Cloud Run service shutdown"
else
    echo "Cloud Run service python-server-api-gcr does not exist."
fi

# Shutdown all Google Compute Engine instances with the service name
echo "Shutting down Compute Engine instances..."
stop_all_instances python-server-api-gcr

# Remove Docker images from Google Container Registry
echo "Removing Docker images from Container Registry..."

delete_image_if_exists() {
    local image_name=$1
    local image_tag=$2

    if gcloud container images list-tags "$image_name" --filter="tags:$image_tag" --format="get(tags)" | grep -q "$image_tag"; then
        gcloud container images delete "$image_name:$image_tag" --quiet
        check_success "Removal of $image_name:$image_tag"
    else
        echo "Image $image_name:$image_tag does not exist."
    fi
}

# Function to check if a Container Registry repository exists
repository_exists() {
    local repo_name=$1
    if gcloud container images list --repository="$repo_name" --format="get(name)" | grep -q "$repo_name"; then
        return 0
    else
        return 1
    fi
}

delete_image_if_exists "gcr.io/$PROJECT_ID/python-server-api-gcr" "latest"
delete_image_if_exists "gcr.io/$PROJECT_ID/ml-instance" "latest"

# Optionally, remove the Container Registry repository if it's empty
echo "Checking if Container Registry repositories are empty..."

if repository_exists "gcr.io/$PROJECT_ID/python-server-api-gcr"; then
    if [ -z "$(gcloud container images list-tags gcr.io/$PROJECT_ID/python-server-api-gcr --format='get(tags)')" ]; then
        gcloud container images delete gcr.io/$PROJECT_ID/python-server-api-gcr --quiet || true
        check_success "Removal of python-server-api-gcr repository"
    else
        echo "python-server-api-gcr repository is not empty."
    fi
else
    echo "Repository python-server-api-gcr does not exist."
fi

if repository_exists "gcr.io/$PROJECT_ID/ml-instance"; then
    if [ -z "$(gcloud container images list-tags gcr.io/$PROJECT_ID/ml-instance --format='get(tags)')" ]; then
        gcloud container images delete gcr.io/$PROJECT_ID/ml-instance --quiet || true
        check_success "Removal of ml-instance repository"
    else
        echo "ml-instance repository is not empty."
    fi
else
    echo "Repository ml-instance does not exist."
fi

echo "Shutdown process complete. All specified cloud resources have been removed."
