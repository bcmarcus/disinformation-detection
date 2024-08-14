import networkx as nx

def time_interval(post_time, event_time):
    return post_time - event_time

def interaction_ratio(post, graph):
    interacting_users = len(post.interacted_users)
    total_users = len(graph.nodes)
    return interacting_users / total_users

def bfs_ratio(post, follower_followee_graph):
    bfs_tree = nx.bfs_tree(follower_followee_graph, post.source_user)
    reachable_nodes = len(bfs_tree.nodes)
    total_nodes = len(follower_followee_graph.nodes)
    return reachable_nodes / total_nodes

def dynamic_attention(post, graph, follower_followee_graph, alpha, beta, gamma):
    time_int = time_interval(post.time, post.event_time)
    int_ratio = interaction_ratio(post, graph)
    bfs_rat = bfs_ratio(post, follower_followee_graph)
    return alpha * time_int + beta * int_ratio + gamma * bfs_rat

def overall_score(post, graph, follower_followee_graph, alpha, beta, gamma, delta):
    attention = dynamic_attention(post, graph, follower_followee_graph, alpha, beta, gamma)
    linguistic_pattern_score = post.linguistic_pattern_score
    return delta * attention + (1 - delta) * linguistic_pattern_score

# Example usage
class Post:
    def __init__(self, time, event_time, source_user, interacted_users, linguistic_pattern_score):
        self.time = time
        self.event_time = event_time
        self.source_user = source_user
        self.interacted_users = interacted_users
        self.linguistic_pattern_score = linguistic_pattern_score

# Create a social network graph (G) and a follower-followee graph (F_G)
G = nx.DiGraph()
F_G = nx.DiGraph()

# Replace the following example with your own data
post = Post(
    time=5,
    event_time=1,
    source_user=1,
    interacted_users={2, 3},
    linguistic_pattern_score=0.6
)

# Add example nodes and edges to the social network graph and follower-followee graph
G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edges_from([(1, 2), (1, 3)])

F_G.add_nodes_from([1, 2, 3, 4, 5])
F_G.add_edges_from([(1, 2), (1, 3)])

# Set weights for the model
alpha = 0.2
beta = 0.3
gamma = 0.4
delta = 0.7

# Calculate the overall score for the post
score = overall_score(post, G, F_G, alpha, beta, gamma, delta)
print("Overall Score:", score)
