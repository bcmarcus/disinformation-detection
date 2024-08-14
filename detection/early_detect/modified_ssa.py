import re
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset
import os

# Text preprocessing
def preprocess(text):
    nlp = spacy.load("en_core_web_sm")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_punct]
    return " ".join(tokens)

# Custom dataset
class PairedDataset(Dataset):
    def __init__(self, tweets, sources, tokenizer):
        self.tweets = tweets
        self.sources = sources
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        positive_source, negative_source = self.sources[idx]
        tweet_encoding = self.tokenizer(tweet, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        pos_source_encoding = self.tokenizer(positive_source, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        neg_source_encoding = self.tokenizer(negative_source, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        
        return tweet_encoding, pos_source_encoding, neg_source_encoding

# Attention mechanism
class MultiLayerAttention(nn.Module):
    def __init__(self, input_size, num_layers):
        super(MultiLayerAttention, self).__init__()
        self.layers = nn.ModuleList([nn.MultiheadAttention(input_size, input_size) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x, x, x)
        return x

# Model
class SemanticSimilarityModel(nn.Module):
    def __init__(self, roberta_model, attention):
        super(SemanticSimilarityModel, self).__init__()
        self.roberta_model = roberta_model
        self.attention = attention

    def forward(self, input_encoding):
        outputs = self.roberta_model(**input_encoding)
        hidden_states = outputs.last_hidden_state
        attention_output = self.attention(hidden_states)
        return attention_output[:, 0]

# Loss function
def contrastive_loss(embeddings_tweet, embeddings_positive, embeddings_negative, alpha=0.5):
    distances_positive = torch.nn.functional.cosine_similarity(embeddings_tweet, embeddings_positive)
    distances_negative = torch.nn.functional.cosine_similarity(embeddings_tweet, embeddings_negative)
    loss = torch.mean(torch.clamp(distances_positive - distances_negative + alpha, min=0))
    return loss

# Read Twitter16 data
def read_twitter16_data(data_dir):
    source_tweets_path = os.path.join(data_dir, "source_tweets.txt")
    labels_path = os.path.join(data_dir, "label.txt")
    
    with open(source_tweets_path, 'r', encoding='utf-8') as f:
        source_tweets = dict(line.strip().split('\t') for line in f)
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = dict(line.strip().split(':') for line in f)
    
    tweets = []
    positive_sources = []
    negative_sources = []
    
    for tweet_id, tweet in source_tweets.items():
        if tweet_id in labels:
            label = labels[tweet_id]
            if label == 'true':
                tweets.append(tweet)
                positive_sources.append(tweet)
            elif label == 'false':
                negative_sources.append(tweet)
    
    # Pair positive and negative sources with tweets
    min_len = min(len(tweets), len(positive_sources), len(negative_sources))
    tweets = tweets[:min_len]
    positive_sources = positive_sources[:min_len]
    negative_sources = negative_sources[:min_len]
    
    sources = list(zip(positive_sources, negative_sources))
    
    return tweets, sources

# Path to Twitter16 data
data_dir = './data/twitter16'

# Load Twitter16 data
tweets, sources = read_twitter16_data(data_dir)

# Preprocess text
tweets = [preprocess(tweet) for tweet in tweets]
sources = [(preprocess(pos), preprocess(neg)) for pos, neg in sources]

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
roberta_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
attention = MultiLayerAttention(roberta_model.config.hidden_size, num_layers=3)
model = SemanticSimilarityModel(roberta_model, attention)

# Training settings
num_epochs = 10
learning_rate = 1e-4
batch_size = 8

# DataLoader
dataset = PairedDataset(tweets, sources, tokenizer)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizer and device
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for tweet_encoding, pos_source_encoding, neg_source_encoding in data_loader:
        # Move inputs to device
        tweet_encoding = {key: value.squeeze(1).to(device) for key, value in tweet_encoding.items()}
        pos_source_encoding = {key: value.squeeze(1).to(device) for key, value in pos_source_encoding.items()}
        neg_source_encoding = {key: value.squeeze(1).to(device) for key, value in neg_source_encoding.items()}
        
        # Forward pass
        embeddings_tweet = model(tweet_encoding)
        embeddings_positive = model(pos_source_encoding)
        embeddings_negative = model(neg_source_encoding)

        # Calculate loss
        loss = contrastive_loss(embeddings_tweet, embeddings_positive, embeddings_negative)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    # Print epoch results
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(data_loader)}")
