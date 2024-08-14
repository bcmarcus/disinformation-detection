import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Set the seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

SEED = 42
set_seed(SEED)

# Load and preprocess the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    # Assuming the dataset has 'text' and 'label' columns
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def tokenize_data(tokenizer, texts, labels, max_length=512):
    tokenized_data = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    input_ids = np.array(tokenized_data['input_ids'])
    attention_masks = np.array(tokenized_data['attention_mask'])
    return input_ids, attention_masks, labels

# Load the dataset and split into train and test sets
file_path = 'OursampleDataset/sample_data.csv'
texts, labels = load_dataset(file_path)
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=SEED)

# Initialize the tokenizer and preprocess the dataset
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_input_ids, train_attention_masks, train_labels = tokenize_data(tokenizer, train_texts, train_labels)
test_input_ids, test_attention_masks, test_labels = tokenize_data(tokenizer, test_texts, test_labels)

# Define the training arguments
output_dir = 'output'
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    seed=SEED,
)

# Load the pre-trained DistilRoBERTa model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(labels)))

# Define the Trainer class instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=list(zip(train_input_ids, train_attention_masks, train_labels)),
    eval_dataset=list(zip(test_input_ids, test_attention_masks, test_labels)),
)

# Start the fine-tuning process
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Save the fine-tuned model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
