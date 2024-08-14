import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv

class UserProfilingModel(nn.Module):
    def __init__(self, input_size, hidden_size, gcn_output_size, ffnn_hidden_size, num_clusters):
        super(UserProfilingModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size)
        self.gcn = GCNConv(hidden_size, gcn_output_size)
        self.ffnn = nn.Sequential(
            nn.Linear(gcn_output_size, ffnn_hidden_size),
            nn.ReLU(),
            nn.Linear(ffnn_hidden_size, 1)
        )
        self.cluster_layer = nn.Linear(gcn_output_size, num_clusters)

    def forward(self, user_profiles, interactions, edge_index):
        _, hidden_states = self.gru(user_profiles)
        user_embeddings = hidden_states[-1]
        transformed_embeddings = self.gcn(user_embeddings, edge_index)
        max_depth = self.ffnn(transformed_embeddings)
        clusters = self.cluster_layer(transformed_embeddings)
        
        return max_depth, clusters

# Example usage
input_size = 10
hidden_size = 32
gcn_output_size = 64
ffnn_hidden_size = 16
num_clusters = 4

user_profiles = torch.randn(5, 10, input_size)  # Replace with actual user profile sequences
interactions = torch.randn(5, input_size)  # Replace with actual user interaction data
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)  # Replace with actual graph connections

model = UserProfilingModel(input_size, hidden_size, gcn_output_size, ffnn_hidden_size, num_clusters)
max_depth, clusters = model(user_profiles, interactions, edge_index)

# Define loss functions
misinfo_loss_fn = nn.BCEWithLogitsLoss()
depth_loss_fn = nn.MSELoss()
cluster_loss_fn = nn.CrossEntropyLoss()

# Define ground truth labels and example outputs
misinfo_labels = torch.tensor([0, 1, 1, 0, 1], dtype=torch.float).unsqueeze(1)
depth_labels = torch.tensor([2.0, 3.0, 1.0, 4.0, 2.0]).unsqueeze(1)
cluster_labels = torch.tensor([0, 1, 2, 3, 1])

# Calculate losses
misinfo_loss = misinfo_loss_fn(max_depth, misinfo_labels)
depth_loss = depth_loss_fn(max_depth, depth_labels)
cluster_loss = cluster_loss_fn(clusters, cluster_labels)

# Define weighting factors
gamma1 = 0.5
gamma2 = 0.3
gamma3 = 0.2

# Calculate total loss
total_loss = gamma1 * misinfo_loss + gamma2 * depth_loss + gamma3 * cluster_loss

# Perform optimization
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()
total_loss.backward()
optimizer.step()

print("Total Loss:", total_loss.item())
