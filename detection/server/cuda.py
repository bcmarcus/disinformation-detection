from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoModelForTokenClassification
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch
import logging
import json
import requests
from bs4 import BeautifulSoup
import re
import ssl
import warnings
import os
from google.cloud import logging as gcloud_logging
from flask import Flask, request, jsonify
from huggingface_hub import login
import random

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.utils")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google Cloud Logging handler
client = gcloud_logging.Client()
gcloud_handler = gcloud_logging.handlers.CloudLoggingHandler(client)
gcloud_handler.setLevel(logging.INFO)

# Add Google Cloud Logging handler to the logger
logger.addHandler(gcloud_handler)

app = Flask(__name__)

# Login to Hugging Face
logger.info("Logging in to Hugging Face...")
login(YOUR_HUGGING_FACE_ACCESS_TOKEN_HERE)

# Load the tokenizer and model for gemma-2-2b-it or gemma-2-9b-it
logger.info("Loading tokenizer and model...")

dev_mode = os.getenv('DEV', 'false').lower() == 'true'

# Set the model name based on the DEV environment variable
if dev_mode:
    logger.info("Using gemma-2-2b-it")
    model_name = "google/gemma-2-2b-it"
else:
    logger.info("Using gemma-2-9b-it")
    model_name = "google/gemma-2-9b-it"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    max_length=4096
)

# Create the transformers pipeline
logger.info("Setting up pipeline...")
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the NER tokenizer and model
tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER", ignore_mismatched_sizes=True)

# Define label mapping for NER
label_list = model_ner.config.id2label
entity_label_mapping = {
    "LOC": "Location",
    "PER": "Person",
    "ORG": "Organization",
    "MISC": "Miscellaneous"
}

def merge_subwords(tokens, labels):
    merged_tokens = []
    merged_labels = []
    current_token = ""
    current_label = ""
    
    for token, label in zip(tokens, labels):
        if token.startswith("##"):
            current_token += token[2:]
        else:
            if current_token:
                merged_tokens.append(current_token)
                merged_labels.append(current_label)
            current_token = token
            current_label = label
    if current_token:
        merged_tokens.append(current_token)
        merged_labels.append(current_label)
    
    return merged_tokens, merged_labels

def ner_process(sentence):
    inputs = tokenizer_ner(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model_ner(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer_ner.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [label_list[label_id.item()] for label_id in predictions[0]]
    
    tokens, labels = merge_subwords(tokens, labels)

    ner_results = []
    current_entity = {"word": "", "entity": ""}

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_entity["word"]:
                ner_results.append(current_entity)
            current_entity = {"word": token, "entity": entity_label_mapping.get(label[2:], label[2:])}
        elif label.startswith("I-") and current_entity["entity"] == entity_label_mapping.get(label[2:], label[2:]):
            current_entity["word"] += " " + token
        else:
            if current_entity["word"]:
                ner_results.append(current_entity)
            current_entity = {"word": "", "entity": ""}

    if current_entity["word"]:
        ner_results.append(current_entity)

    return ner_results

def get_first_non_disambiguation_link(title):
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        for li in soup.select('.mw-parser-output li'):
            a = li.find('a')
            if a and not a.get('href').startswith('/wiki/Help:') and not 'may refer to:' in li.get_text():
                new_title = a['href'].split('/')[-1]
                # description, article_content, link = get_article_content_and_link(new_title.replace('_', ' '))
                # return description, link, article_content
                return new_title.replace('_', ' ')
    return "No description found", "No article content found", None

def get_wikipedia_content(title):
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.find(id="mw-content-text").find(class_="mw-parser-output")
        
        for unwanted in content(["table", "div", "script", "style"]):
            unwanted.decompose()
        
        text = ' '.join(p.text for p in content.find_all('p'))
        text = re.sub(r'\[[\d\s]+\]', '', text)
        return text, url
    return "", url

def get_article_content_and_link(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|description",
        "explaintext": True,
        "redirects": 1,
        "format": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    if pages:
        for page_id, page_data in pages.items():
            extract = page_data.get("extract", "")
            description = page_data.get("description", "")
            link = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            return description, extract, link
    return "No description found", "No article content found", None

def search_wikipedia(query):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    # if data.get("query", {}).get("search", []):
    #     first_result_title = data["query"]["search"][0]["title"]
    #     return get_wikipedia_content(first_result_title)
    # return "", ""
    if data.get("query", {}).get("search", []):
        first_result_title = data["query"]["search"][0]["title"]
        description, article_content, link = get_article_content_and_link(first_result_title)
        
        # Check if it's a disambiguation page
        if "disambiguation" in description.lower() or "refer to:" in article_content.lower():
            first_result_title = get_first_non_disambiguation_link(first_result_title)
        
        return get_wikipedia_content(first_result_title)
    else:
        return "No information found", None

def create_vector_store(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    documents = text_splitter.create_documents(texts)
    
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def get_answer(question, context):
    template = """<bos><start_of_turn>user
    Use the following pieces of context to determine the veracity of the statement and explain why in a concise manner, in 3 sentences.
    The answer should be one of five possible outputs. True, Mostly True, Somewhat True, Mostly False, False or Not Applicable.
    Use the format Answer: *the actual answer* Explanation: *the actual explanation*.

    Context:
    {context}

    Statement: {question}<end_of_turn>
    <start_of_turn>model
    """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=context.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    logger.info("Invoking the QA Chain")

    result = qa_chain.invoke({"query": question})
    return result["result"]

def parse_output(output):
    # Match the model's response part only, ignoring the prompt part
    model_response_pattern = re.compile(r"<start_of_turn>model\s*(.*)", re.DOTALL)
    model_response_match = model_response_pattern.search(output)
    
    if not model_response_match:
        return "No answer found", "No explanation found"
    
    model_response = model_response_match.group(1).strip()

    # Now extract the answer and explanation from the model's response
    answer_pattern = re.compile(r"Answer:\s*(.*?)\s*Explanation:", re.DOTALL)
    explanation_pattern = re.compile(r"Explanation:\s*(.*)", re.DOTALL)

    answer_match = answer_pattern.search(model_response)
    explanation_match = explanation_pattern.search(model_response)
    
    answer = answer_match.group(1).strip() if answer_match else "No answer found"
    explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found"

    return answer, explanation

def get_query_input(text):
    # Extract the title using a regular expression
    title_match = re.search(r"Title: (.+?)\\nPost:", text)
    
    if title_match:
        title = title_match.group(1)
        input_text = title
    else:
        # Remove the 'Title: ' part if it exists and limit to 20 words
        content = re.sub(r"Title: ", "", text)
        words = content.split()
        input_text = ' '.join(words[:20])
    
    return input_text

def process_input(input_text):
    logger.info(f"Processing input: {input_text}")
    
    # Perform NER on the input
    entities = ner_process(input_text)
    logger.info("NER Results:")
    for entity in entities:
        logger.info(f"- {entity['word']} ({entity['entity']})")
    
    # Collect Wikipedia content for each entity and the full query
    wiki_contents = []
    retrieved_articles = set()
    article_links = []
    
    # Search for the full query
    full_query_input = get_query_input(input_text)
    print ("FULL_INPUT:" + full_query_input)
    full_query_content, full_query_link = search_wikipedia(full_query_input)
    if full_query_content and full_query_content not in retrieved_articles:
        wiki_contents.append(full_query_content)
        retrieved_articles.add(full_query_content)
        article_links.append(full_query_link)
        logger.info(f"Retrieved article for full query '{input_text}'. Word count: {len(full_query_content.split())}")
    elif full_query_content in retrieved_articles:
        logger.info(f"Article for '{input_text}' already utilized")
    else:
        logger.info(f"No article found for full query '{input_text}'")
    
    # Search for individual entities
    for entity in entities:
        content, link = search_wikipedia(entity["word"])
        if content and content not in retrieved_articles:
            wiki_contents.append(content)
            retrieved_articles.add(content)
            article_links.append(link)
            logger.info(f"Retrieved article for '{entity['word']}'. Word count: {len(content.split())}")
        elif content in retrieved_articles:
            logger.info(f"Article for '{entity['word']}' already utilized")
        else:
            logger.info(f"No article found for '{entity['word']}'")
    
    # Create vector store from Wikipedia content
    if wiki_contents:
        vector_store = create_vector_store(wiki_contents)
    else:
        # If no Wikipedia content found, create an empty vector store
        vector_store = FAISS.from_texts(["No relevant information found."], embeddings)
    
    # Get the answer
    answer = get_answer(input_text, vector_store)
    
    print (answer)
    # Parse the output to get the answer and explanation
    parsed_answer, parsed_explanation = parse_output(answer)
    
    # Format the result to include Wikipedia article links
    formatted_result = {
        "score": random.uniform(0, 100),
        "answer": parsed_answer,
        "explanation": parsed_explanation,
        "links": [{"title": link.split('/')[-1].replace('_', ' '), "link": link} for link in article_links if link]
    }

    
    return formatted_result

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("ML Instance health check requested")
    return jsonify({"status": "200"}), 200
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        logger.warning("Predict endpoint received no JSON data")
        return jsonify({"error": "No JSON data provided"}), 400

    logger.info(f'Data received: {data}')
    input_text = data.get("input")

    if not input_text:
        logger.warning("Predict endpoint received no input text")
        return jsonify({"error": "No input text provided"}), 400

    try:
        result = process_input(input_text)
        logger.info(f"Process response: {result}")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred during processing"}), 500

@app.route('/isFalse', methods=['POST'])
def is_false():
    logger.info("isFalse endpoint requested")
    data = request.json
    if not data or 'post' not in data:
        logger.warning("isFalse endpoint received invalid request")
        return jsonify({"error": "Invalid request. 'post' field is required."}), 400

    input_text = data['post']
    try:
        result = process_input(input_text)
        logger.info(f"isFalse response: {result}")
        
        # Determine if the statement is false based on the response
        is_false = "false" in result["answer"].lower() or "not true" in result["answer"].lower()
        
        response = {
            "is_false": is_false,
            "explanation": result["explanation"]
        }
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error processing isFalse request: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred during processing"}), 500

if __name__ == '__main__':
    logger.info("Starting ML server...")
    app.run(host='0.0.0.0', port=5000)
