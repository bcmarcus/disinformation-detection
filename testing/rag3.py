import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForTokenClassification
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import re
import ssl
import warnings

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.utils")

# Load the tokenizer and model for microsoft/Phi-3-mini-4k-instruct
# model = "microsoft/Phi-3-mini-128k-instruct"
model = "microsoft/phi-2"
tokenizer_llm = AutoTokenizer.from_pretrained(model)
model_llm = AutoModelForCausalLM.from_pretrained(
    model,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    max_length=1024
)

# Create the transformers pipeline for LLM
hf_pipeline = pipeline(
    "text-generation",
    model=model_llm,
    tokenizer=tokenizer_llm,
)

# Create a LangChain LLM wrapper
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
        return text
    return ""

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
    if data.get("query", {}).get("search", []):
        first_result_title = data["query"]["search"][0]["title"]
        return get_wikipedia_content(first_result_title)
    return ""

def create_vector_store(texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.create_documents(texts)
    
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def get_answer(question, context):
    template = """Use the following pieces of context to determine if the statement is true or false, and explain why. 
    If you don't have enough information, say so and explain why.

    Context:
    {context}

    Statement: {question}
    Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=context.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    result = qa_chain.invoke({"query": question})
    return result["result"]

def process_input(input_text):
    print(f"Input: {input_text}")
    
    # Perform NER on the input
    entities = ner_process(input_text)
    print("NER Results:")
    for entity in entities:
        print(f"- {entity['word']} ({entity['entity']})")
    
    # Collect Wikipedia content for each entity and the full query
    wiki_contents = []
    
    # Search for the full query
    full_query_content = search_wikipedia(input_text)
    if full_query_content:
        wiki_contents.append(full_query_content)
        print(f"Retrieved article for full query '{input_text}'. Word count: {len(full_query_content.split())}")
    else:
        print(f"No article found for full query '{input_text}'")
    
    # Search for individual entities
    for entity in entities:
        content = search_wikipedia(entity["word"])
        if content:
            wiki_contents.append(content)
            print(f"Retrieved article for '{entity['word']}'. Word count: {len(content.split())}")
        else:
            print(f"No article found for '{entity['word']}'")
    
    # Create vector store from Wikipedia content
    if wiki_contents:
        vector_store = create_vector_store(wiki_contents)
    else:
        # If no Wikipedia content found, create an empty vector store
        vector_store = FAISS.from_texts(["No relevant information found."], embeddings)
    
    # Get the answer
    answer = get_answer(input_text, vector_store)
    
    return answer

# Example usage
test_cases = [
    "My name is Wolfgang and I live in Berlin.",
    "Google was founded by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University.",
    "The Eiffel Tower is located in Paris.",
    "Amazon was started in Seattle by Jeff Bezos.",
    "Barack Obama was the 44th President of the United States.",
    "Trump doesn't belive in Covid-19.",
    "The Earth is flat."
]

input_text = "Trump doesn't belive in Covid-19."
result = process_input(input_text)
print(f"\nResult: {result}")