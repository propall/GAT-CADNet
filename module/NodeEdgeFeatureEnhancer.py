import torch
import torch.nn as nn
import torch.nn.functional as F


# Used to feature extraction of node features and edge features
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class NodeEdgeFeatureEnhancer(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim):
        super().__init__()

        # MLP extracts node features and edge features
        self.node_mlp = MLP(node_input_dim, 64)     # Map node features to 64-dimensional
        self.edge_mlp = MLP(edge_input_dim, 64)     # Map edge features to 64-dimensional

        # Final output dimension
        self.output_dim = output_dim

    def forward(self, node_features, edge_features):
        # Embed node features
        node_emb = self.node_mlp(node_features)     # Node embedding v_hat_i

        # Embed edge features
        edge_emb = self.edge_mlp(edge_features)     # Edge embed e_hat_ij

        # Pool edge features of each node(max pooling)
        if edge_emb.size(0) > 0:
            pooled_edge_feats = torch.max(edge_emb, dim=0)[0].unsqueeze(0)
        else:
            pooled_edge_feats = torch.zeros(edge_emb.size(1), device=node_features.device).unsqueeze(0)  # Processing empty

        # Spliced ​​node features and edge features after pooling
        enhanced_node_features = torch.cat([node_emb, pooled_edge_feats], dim=1)

        return enhanced_node_features

