import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets, DatasetDict

# Load and combine datasets
def load_and_combine_datasets():
    datasets = []
    data_paths = ['data/ANTIVax', 'data/covid19-fake-news-detection', 'data/Extended-Pheme-Dataset', 
                  'data/FakeNewsNet', 'data/OursampleDataset', 'data/twitter15', 'data/twitter16', 'data/fakeddit']
    
    for path in data_paths:
        data = load_dataset('csv', data_files={'train': f'{path}/train.csv', 'test': f'{path}/test.csv'})
        datasets.append(data)
    
    train_datasets = [d['train'] for d in datasets]
    test_datasets = [d['test'] for d in datasets]
    
    combined_train = concatenate_datasets(train_datasets)
    combined_test = concatenate_datasets(test_datasets)
    
    return DatasetDict({'train': combined_train, 'test': combined_test})

combined_datasets = load_and_combine_datasets()

# Tokenize datasets
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = combined_datasets.map(tokenize_function, batched=True)

# Load model and training arguments
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

# Fine-tune model
trainer.train()

# Evaluate and save model
eval_results = trainer.evaluate()
print(eval_results)

model.save_pretrained('path/to/save/fine-tuned-model')
tokenizer.save_pretrained('path/to/save/fine-tuned-model')
