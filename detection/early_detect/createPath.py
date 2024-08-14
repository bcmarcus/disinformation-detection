import pandas as pd
import networkx as nx

# Load labelled input from datasets
true_df = pd.read_csv('true_dataset.csv')
false_df = pd.read_csv('false_dataset.csv')

# Create full network using the create_full_network function
G = create_full_network(api_key, api_secret_key, access_token, access_token_secret, true_df['tweet_id'].tolist() + false_df['tweet_id'].tolist())

# Create two different propagation paths for true and false information flow
true_path = nx.DiGraph()
false_path = nx.DiGraph()

for node in G.nodes:
    if node in true_df['tweet_id'].tolist():
        true_path.add_node(node)
        predecessors = list(G.predecessors(node))
        for predecessor in predecessors:
            if predecessor in true_df['user_id'].tolist():
                true_path.add_node(predecessor)
                true_path.add_edge(predecessor, node)
    elif node in false_df['tweet_id'].tolist():
        false_path.add_node(node)
        predecessors = list(G.predecessors(node))
        for predecessor in predecessors:
            if predecessor in false_df['user_id'].tolist():
                false_path.add_node(predecessor)
                false_path.add_edge(predecessor, node)
