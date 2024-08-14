import networkx as nx

def personalized_page_rank(G, credible_users, alpha=0.85, max_iter=100, tol=1.0e-6):
    personalization = {node: 1 if node in credible_users else 0 for node in G.nodes()}
    trust_scores = nx.pagerank(G, alpha=alpha, personalization=personalization, max_iter=max_iter, tol=tol)
    return trust_scores

def update_edge_weights(G, trust_scores):
    for u, v, edge_data in G.edges(data=True):
        edge_type = edge_data['edge_type']
        T_u = trust_scores[u]
        T_v = trust_scores[v]
        
        if edge_type == 'RT':  # Retweet
            NRT = edge_data['NRT']
            G[u][v]['weight'] = T_u * T_v * NRT
        elif edge_type == 'M':  # Mention
            NM = edge_data['NM']
            G[u][v]['weight'] = T_u * T_v * NM
        elif edge_type == 'F':  # Follow
            G[u][v]['weight'] = T_u * T_v

def early_misinformation_detection(graph, credible_users):
    trust_scores = personalized_page_rank(graph, credible_users)
    update_edge_weights(graph, trust_scores)
    return graph

# Example usage
G = nx.DiGraph()
credible_users = set(['user1', 'user2'])

# Add nodes and edges to the graph
G.add_edge('user1', 'user2', edge_type='RT', NRT=1)
G.add_edge('user2', 'user3', edge_type='M', NM=2)
G.add_edge('user1', 'user3', edge_type='F')

updated_graph = early_misinformation_detection(G, credible_users)
