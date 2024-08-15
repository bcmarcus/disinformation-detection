import json
import os
import random
from flask import Flask, request, jsonify
import requests
from google.cloud import compute_v1, logging as gcloud_logging
from google.auth import default
import logging
from logging.handlers import RotatingFileHandler
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure local file handler
file_handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
file_handler.setLevel(logging.DEBUG)

# Configure Google Cloud Logging handler
client = gcloud_logging.Client()
gcloud_handler = gcloud_logging.handlers.CloudLoggingHandler(client)
gcloud_handler.setLevel(logging.INFO)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(gcloud_handler)

logger.info("Starting application...")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

DEV_MODE = os.environ.get('DEV', 'false').lower() == 'true'

if DEV_MODE:
    logger.info("Running in development mode")
    LLM_HANDLER_URL = os.environ.get('LLM_HANDLER_URL', 'http://localhost:5000')
else:
    logger.info("Running in production mode")
    
    # Use environment variable for LLM handler URL
    LLM_HANDLER_URL = os.environ.get('LLM_HANDLER_URL')

    if not LLM_HANDLER_URL:
        logger.error("LLM_HANDLER_URL environment variable is not set")
        raise ValueError("LLM_HANDLER_URL environment variable must be set")

    logger.info(f"LLM Handler URL: {LLM_HANDLER_URL}")

def backup_prediction():
    singleLinkConfig = [
      {
          "title": "Learn More",
          "description": "Get more information from Wikipedia",
          "link": "https://www.wikipedia.org/"
      }
    ]
    doubleLinkConfig = [
      {
          "title": "COVID-19",
          "description": "Get the latest information from the MOH about coronavirus.",
          "link": "https://www.wikipedia.org/"
      },
      {
          "title": "More Resources",
          "description": "See more resources on Google",
          "link": "https://www.wikipedia.org/"
      }
    ]

    return {
        "score": random.uniform(0, 100),
        "answer": "Some likelihood of truth",
        "explanation": "Some explanation",
        "links": random.choice([singleLinkConfig, doubleLinkConfig])
    }

def get_score(answer):
    score_map = {
        "True": 100,
        "Mostly True": 75,
        "Somewhat True": 50,
        "Mostly False": 25,
        "False": 0,
        "Not Applicable": 100
    }
    return score_map.get(answer, 0)  # Default to 0 if answer is not in the map

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Run health check requested")
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
    logger.info(f"Predict endpoint requested to {LLM_HANDLER_URL}/predict")
    data = request.json
    if not data:
        logger.warning("Predict endpoint received no JSON data")
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        response = requests.post(f"{LLM_HANDLER_URL}/predict", json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # Get the score based on the answer
        if 'answer' in result:
            score = get_score(result['answer'])
            result['score'] = score
            logger.info(f"Assigned score {score} based on answer {result['answer']}")
        else:
            logger.warning("No 'answer' field found in LLM handler response")

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

    try:
        response = requests.post(f"{LLM_HANDLER_URL}/isFalse", json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        logger.info(f"isFalse endpoint result: {result}")
        return jsonify(result), 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in isFalse endpoint: {str(e)}", exc_info=True)
        random_number = random.randint(0, 100)
        logger.info(f"Using backup random number: {random_number}")
        return jsonify({"random_number": random_number}), 200

if __name__ == '__main__':
    logger.info("Starting Flask server")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
