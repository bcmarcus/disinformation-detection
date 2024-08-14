from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import ssl

# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

# Define label mapping
label_list = model.config.id2label

# Define test cases
test_cases = [
    "My name is Wolfgang and I live in Berlin.",
    "Google was founded by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University.",
    "The Eiffel Tower is located in Paris.",
    "Amazon was started in Seattle by Jeff Bezos.",
    "Barack Obama was the 44th President of the United States."
]

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

# Process and print results for each test case
for i, sentence in enumerate(test_cases):
    ner_results = ner_process(sentence)
    print(f"Test Case {i+1}: {sentence}")
    print(ner_results)
    print("\n")
