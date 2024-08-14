import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import spacy
import re
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

# 1. Data Parsing and Preprocessing
def load_and_preprocess_data(file_path, dataset_type):
    if dataset_type == 'reddit':
        df = pd.read_csv(file_path, sep='\t')
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
        df['content'] = df['clean_title'].fillna('') + ' ' + df['title'].fillna('')
        df['label'] = df['2_way_label']
        return df[['author', 'content', 'created_utc', 'subreddit', 'label']]
    elif dataset_type == 'twitter':
        # Twitter data loading remains unchanged
        labels = pd.read_csv(file_path + '/label.txt', sep=':', header=None, names=['label', 'tweet_id'])
        tweets = pd.read_csv(file_path + '/source_tweets.txt', sep='\t', header=None, names=['tweet_id', 'content'])
        df = pd.merge(labels, tweets, on='tweet_id')
        df['created_utc'] = pd.to_datetime('now')  # As we don't have actual timestamps
        df['author'] = 'unknown'  # As we don't have author information
        df['subreddit'] = 'twitter'  # Use 'twitter' as a placeholder
        return df[['author', 'content', 'created_utc', 'subreddit', 'label', 'tweet_id']]

# 2. Network Construction
def construct_network(df, dataset_type):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['author'], content=row['content'], label=row['label'])
    
    if dataset_type == 'reddit':
        subreddit_users = df.groupby('subreddit')['author'].apply(list)
        for users in subreddit_users:
            for i in range(len(users)):
                for j in range(i+1, len(users)):
                    G.add_edge(users[i], users[j])
    elif dataset_type == 'twitter':
        # Twitter network construction remains unchanged
        for tweet_id in df['tweet_id']:
            try:
                with open(f"{file_path}/tree/{tweet_id}.txt", 'r') as f:
                    tree = f.readlines()
                if len(tree) > 5:
                    for line in tree[1:]:  # Skip the root
                        parent, child = line.strip().split('->')
                        parent_id = eval(parent)[1]
                        child_id = eval(child)[1]
                        G.add_edge(parent_id, child_id)
                else:
                    # If less than 5 comments, connect all nodes
                    nodes = [eval(line.split('->')[1])[1] for line in tree]
                    for i in range(len(nodes)):
                        for j in range(i+1, len(nodes)):
                            G.add_edge(nodes[i], nodes[j])
            except FileNotFoundError:
                print(f"Tree file for tweet {tweet_id} not found.")
    
    return G


# 3. User Profiling
class UserProfileDataset(Dataset):
    def __init__(self, graph, tokenizer, max_length=128):
        self.graph = graph
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.users = list(graph.nodes())

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        content = self.graph.nodes[user]['content']
        label = self.graph.nodes[user]['label']
        
        encoding = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 4. Causal User Attribute Inference (CUAI)
class CUAI(nn.Module):
    def __init__(self, pretrained_model):
        super(CUAI, self).__init__()
        self.roberta = pretrained_model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.roberta.config.hidden_size, 2)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

# 5. Temporal Characteristics
def calculate_temporal_features(df):
    df['time_delta'] = (df['created_utc'] - df['created_utc'].min()).dt.total_seconds()
    df['interaction_ratio'] = df.groupby('author').cumcount() / len(df)
    df['bfs_ratio'] = df.groupby('author').cumcount() / df['author'].nunique()
    return df

def time_interval(post_time, event_time):
    return (post_time - event_time).total_seconds()

def interaction_ratio(post, graph):
    interacting_users = len(list(graph.neighbors(post['author'])))
    total_users = len(graph.nodes)
    return interacting_users / total_users

def bfs_ratio(post, graph):
    bfs_tree = nx.bfs_tree(graph, post['author'])
    reachable_nodes = len(bfs_tree.nodes)
    total_nodes = len(graph.nodes)
    return reachable_nodes / total_nodes

def dynamic_attention(post, graph, alpha, beta, gamma):
    time_int = time_interval(post['created_utc'], post['created_utc'].min())
    int_ratio = interaction_ratio(post, graph)
    bfs_rat = bfs_ratio(post, graph)
    return alpha * time_int + beta * int_ratio + gamma * bfs_rat

# 6. Linguistic Pattern Analysis
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_punct]
    return " ".join(tokens)

class SemanticSimilarityAnalysis(nn.Module):
    def __init__(self, roberta_model, num_attention_layers=3):
        super(SemanticSimilarityAnalysis, self).__init__()
        self.roberta = roberta_model
        self.attention_layers = nn.ModuleList([nn.MultiheadAttention(768, 8) for _ in range(num_attention_layers)])
        self.fc = nn.Linear(768, 768)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        for layer in self.attention_layers:
            hidden_states, _ = layer(hidden_states, hidden_states, hidden_states)
        return self.fc(hidden_states[:, 0, :])

def contrastive_loss(embeddings_tweet, embeddings_positive, embeddings_negative, alpha=0.5):
    distances_positive = torch.nn.functional.cosine_similarity(embeddings_tweet, embeddings_positive)
    distances_negative = torch.nn.functional.cosine_similarity(embeddings_tweet, embeddings_negative)
    loss = torch.mean(torch.clamp(distances_positive - distances_negative + alpha, min=0))
    return loss

def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# 7. Model Training and Evaluation
def train_and_evaluate(model, train_dataloader, val_dataloader, device, epochs=3):
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(classification_report(val_labels, val_preds))

# Main execution
def run_cipher(file_path, dataset_type):
    # Load and preprocess data
    df = load_and_preprocess_data(file_path, dataset_type)
    
    # Construct network
    G = construct_network(df, dataset_type)
    
    # Calculate temporal features
    df = calculate_temporal_features(df)
    
    # Add linguistic features
    df['preprocessed_content'] = df['content'].apply(preprocess_text)
    df['sentiment_polarity'], df['sentiment_subjectivity'] = zip(*df['preprocessed_content'].apply(sentiment_analysis))
    
    # Prepare for model training
    dataset = UserProfileDataset(G, tokenizer)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Initialize and train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CUAI(roberta_model).to(device)
    
    train_and_evaluate(model, train_dataloader, val_dataloader, device)

    # Initialize and train SSA model
    ssa_model = SemanticSimilarityAnalysis(roberta_model).to(device)
    # Train SSA model (implementation omitted for brevity)

    return model, ssa_model, G, df

if __name__ == "__main__":
    # Example usage
    reddit_model, reddit_ssa, reddit_graph, reddit_df = run_cipher('./data/Fakeddit/all_samples/all_train.tsv', 'reddit')
    # twitter_model, twitter_ssa, twitter_graph, twitter_df = run_cipher('./data/twitter_data', 'twitter')