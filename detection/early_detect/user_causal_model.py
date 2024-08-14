import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data

class Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class CUAIModel(nn.Module):
    def __init__(self, num_features, embedding_dim, num_heads, num_gat_layers):
        super(CUAIModel, self).__init__()
        self.attribute_embedding = nn.Linear(num_features, embedding_dim)
        self.gat_layers = nn.ModuleList([gnn.GATConv(embedding_dim, embedding_dim, heads=num_heads, concat=True) for _ in range(num_gat_layers)])
        self.regression_y1 = Regression(embedding_dim * num_heads, 1)
        self.regression_y0 = Regression(embedding_dim * num_heads, 1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * num_heads, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        h = self.attribute_embedding(x)
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index)
        y1 = self.regression_y1(h)
        y0 = self.regression_y0(h)
        causal_effect = y1 - y0
        propensity = self.classifier(h)
        return causal_effect, propensity

# Example usage
num_users = 5
num_features = 5
embedding_dim = 16
num_heads = 8
num_gat_layers = 2

# User attributes (A(v))
attributes = torch.randn(num_users, num_features)

# Define the edges of the social network graph
edge_index = torch.tensor([
    [0, 1, 2, 0, 3],
    [1, 0, 1, 3, 4]
], dtype=torch.long)

# Create a PyTorch Geometric graph data object
data = Data(x=attributes, edge_index=edge_index)

model = CUAIModel(num_features=num_features, embedding_dim=embedding_dim, num_heads=num_heads, num_gat_layers=num_gat_layers)
causal_effects, propensities = model(data.x, data.edge_index)
print("Causal Effects:", causal_effects)
print("Propensities:", propensities)
