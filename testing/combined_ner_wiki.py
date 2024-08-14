import ssl
import torch
import requests
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

# Define label mapping
label_list = model.config.id2label

# Define a mapping for more descriptive labels
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
    # Tokenize and get model predictions
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted labels
    predictions = torch.argmax(outputs.logits, dim=2)

    # Decode tokens and labels
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [label_list[label_id.item()] for label_id in predictions[0]]
    
    # Merge subword tokens
    tokens, labels = merge_subwords(tokens, labels)

    # Combine tokens and labels into full entities
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

# Define the order of importance
importance_order = ["Miscellaneous", "Person", "Organization", "Location"]
NUM_ENTITIES = 2

def get_top_entities(entities):
    sorted_entities = sorted(entities, key=lambda item: importance_order.index(item['entity']))
    return sorted_entities[:NUM_ENTITIES]

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
        description, article_content, link = get_article_content_and_link(first_result_title)
        
        # Check if it's a disambiguation page
        if "disambiguation" in description.lower() or "refer to:" in article_content.lower():
            return get_first_non_disambiguation_link(first_result_title)
        
        return description, link, article_content
    else:
        return "No information found", None, "No article content found"

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
                description, article_content, link = get_article_content_and_link(new_title.replace('_', ' '))
                return description, link, article_content
    return "No description found", "No article content found", None

def get_wikipedia_info(entities):
    top_entities = get_top_entities(entities)
    info = {}
    for entity in top_entities:
        entity_value = entity["word"]
        description, link, article_content = search_wikipedia(entity_value)
        info[entity_value] = {
            "snippet": description,
            "link": link,
            "description": description,
            "full_article": article_content
        }
    return info

# Define test cases
test_cases = [
    "My name is Wolfgang and I live in Berlin.",
    "Google was founded by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University.",
    "The Eiffel Tower is located in Paris.",
    "Amazon was started in Seattle by Jeff Bezos.",
    "Barack Obama was the 44th President of the United States.",
    "Trump doesn't belive in Covid-19."
]

# Process and print results for each test case
for i, sentence in enumerate(test_cases):
    ner_results = ner_process(sentence)
    print(f"Test Case {i+1}: {sentence}")
    print(ner_results)
    print(f"Entities extracted: {ner_results}")
    print(f"Test Set {i+1}:")
    if not ner_results:
        print("No entities to search.")
    else:
        info = get_wikipedia_info(ner_results)
        for entity, details in info.items():
            print(f"{entity}: {details['snippet']}")
            if details['link']:
                print(f"Link: {details['link']}")
                print(f"Description: {details['description']}")
                print(f"Full article (first 200 chars): {details['full_article'][:200]}")
            print(f"Full article length: {len(details['full_article'])} characters")
    print("\n" + "-"*40 + "\n")
