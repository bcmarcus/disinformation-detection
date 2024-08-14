import requests
import json
import os

BASE_URL = os.environ.get('SERVER_URL', 'http://localhost:8080')

def test_health_check():
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("Health Check Status Code:", response.status_code)
        print("Health Check Response:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Health Check failed: {e}")

def test_json_endpoint():
    test_data = {
        "name": "John Doe",
        "age": 30,
        "city": "New York"
    }
    try:
        response = requests.post(f"{BASE_URL}/test", json=test_data)
        print("Test Endpoint Status Code:", response.status_code)
        print("Test Endpoint Response:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Test Endpoint failed: {e}")

def test_predict_endpoint():
    predict_data = {
        "input": "What is the weather like today? If you don't know just make something up.",
        "type": "formatted"
    }
    try:
        response = requests.post(f"{BASE_URL}/predict", json=predict_data)
        print("Predict Endpoint (Formatted) Status Code:", response.status_code)
        print("Predict Endpoint (Formatted) Response:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Predict Endpoint (Formatted) failed: {e}")

    predict_data["type"] = "plain"
    try:
        response = requests.post(f"{BASE_URL}/predict", json=predict_data)
        print("Predict Endpoint (Plain) Status Code:", response.status_code)
        print("Predict Endpoint (Plain) Response:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Predict Endpoint (Plain) failed: {e}")

def test_is_false_endpoint():
    correct_data = {
        "post": {
            "subfield1": "value1",
            "subfield2": "value2"
        }
    }
    incorrect_data = {
        "post": {
            "subfield1": "value1"
        },
        "extra_field": "not allowed"
    }
    try:
        response = requests.post(f"{BASE_URL}/isFalse", json=correct_data)
        print("isFalse Endpoint (Correct Data) Status Code:", response.status_code)
        print("isFalse Endpoint (Correct Data) Response:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"isFalse Endpoint (Correct Data) failed: {e}")

    try:
        response = requests.post(f"{BASE_URL}/isFalse", json=incorrect_data)
        print("isFalse Endpoint (Incorrect Data) Status Code:", response.status_code)
        print("isFalse Endpoint (Incorrect Data) Response:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"isFalse Endpoint (Incorrect Data) failed: {e}")

if __name__ == "__main__":
    test_health_check()
    test_json_endpoint()
    test_predict_endpoint()
    test_is_false_endpoint()
