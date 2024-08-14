import json
import os
import random
from flask import Flask, request, jsonify
import requests
from google.cloud import compute_v1
from google.auth import default
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

DEV_MODE = os.environ.get('DEV', 'false').lower() == 'true'

if DEV_MODE:
    logger.info("Running in development mode")
    LLM_HANDLER_URL = os.environ.get('LLM_HANDLER_URL', 'http://localhost:8001')
else:
    logger.info("Running in production mode")
    PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT')
    ZONE = os.environ.get('GOOGLE_CLOUD_ZONE', 'us-central1-a')
    INSTANCE_NAME = os.environ.get('GCE_INSTANCE_NAME', 'ml-instance')

    def get_gce_instance_ip():
        credentials, project = default()
        instance_client = compute_v1.InstancesClient(credentials=credentials)
        instance = instance_client.get(project=PROJECT_ID, zone=ZONE, instance=INSTANCE_NAME)
        return instance.network_interfaces[0].access_configs[0].nat_i_p

    GCE_IP = get_gce_instance_ip()
    LLM_HANDLER_URL = f'http://{GCE_IP}:8001'

def backup_prediction():
    return {
        "score": random.uniform(0, 100),
        "message": "Hello, World!"
    }

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check requested")
    return jsonify({"status": "200"}), 200

@app.route('/test', methods=['POST'])
def test_endpoint():
    logger.info("Test endpoint requested")
    data = request.json
    if data:
        top_level_keys = list(data.keys())
        logger.info(f"Test endpoint received data with keys: {top_level_keys}")
        return jsonify(data), 200
    else:
        logger.warning("Test endpoint received no JSON data")
        return jsonify({"error": "No JSON data provided"}), 400

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Predict endpoint requested")
    data = request.json
    if not data:
        logger.warning("Predict endpoint received no JSON data")
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        response = requests.post(f"{LLM_HANDLER_URL}/predict", json=data)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Predict endpoint result: {result}")
        return jsonify(result), 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
        result = backup_prediction()
        logger.info(f"Using backup prediction: {result}")
        return jsonify(result), 200

@app.route('/isFalse', methods=['POST'])
def is_false():
    logger.info("isFalse endpoint requested")
    data = request.json
    if not data or 'post' not in data or len(data) != 1:
        logger.warning("isFalse endpoint received invalid JSON data")
        return jsonify({"error": "Invalid JSON data provided. Must contain only 'post' field."}), 400
    
    random_number = random.randint(0, 100)
    logger.info(f"isFalse endpoint generated random number: {random_number}")
    return jsonify({"random_number": random_number}), 200

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
