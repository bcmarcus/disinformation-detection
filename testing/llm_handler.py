from flask import Flask, request, jsonify
from huggingface_hub import login
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pydantic import BaseModel
import torch
import logging
import json
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Login to Hugging Face
login(YOUR_HUGGING_FACE_ACCESS_TOKEN_HERE)

# Define the output schema using pydantic
class Weather(BaseModel):
    weather: str
    chance_of_rain: float

# Load the tokenizer and model for microsoft/Phi-3-mini-4k-instruct
logger.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    max_length=200
)
logger.info("Tokenizer and model loaded successfully.")

# Create the transformers pipeline
logger.info("Setting up pipeline...")
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
logger.info("Pipeline setup complete.")

# Create a character-level parser and build a transformers prefix function from it
parser = JsonSchemaParser(Weather.schema())
prefix_function = build_transformers_prefix_allowed_tokens_fn(hf_pipeline.tokenizer, parser)

# Define chat structure
system_message = {"role": "system", "content": "You are a helpful AI assistant."}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    logger.info(f'Data received: {data}')
    input_text = data.get("input")
    request_type = data.get("type")

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    try:
        if request_type == "formatted":
            logger.info("Processing formatted request...")
            schema = json.loads(Weather.schema_json())
            properties_only = {key: {"type": value["type"]} for key, value in schema["properties"].items()}
            prompt = f"Answer this question. Keep the json output short. {input_text}\nProvide a response following the format of this JSON schema: {properties_only}\n"
            generation_args = {
                "max_new_tokens": 500,
                "return_full_text": False,
                "temperature": 0.05,
                "do_sample": True,
                "prefix_allowed_tokens_fn": prefix_function,
            }
        else:
            logger.info("Processing plain request...")
            prompt = f"Answer this question in one sentence. Keep it short. After you have answered the question DO NOT SAY ANYTHING ELSE. {input_text}\n"
            generation_args = {
                "max_new_tokens": 500,
                "return_full_text": False,
                "temperature": 0.05,
                "do_sample": True,
            }

        messages = [
            system_message,
            {"role": "user", "content": prompt}
        ]

        output = hf_pipeline(messages, **generation_args)
        result = output[0]['generated_text']
        logger.info(f"Chain response: {result}")

        if request_type == "formatted":
            try:
                formatted_response = json.loads(result)
                response = {"response": formatted_response}
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, returning raw result")
                response = {"response": result}
        else:
            response = {"response": result}

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/isFalse', methods=['POST'])
def is_false():
    data = request.json
    random_number = random.randint(0, 100)
    return jsonify({"random_number": random_number}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
