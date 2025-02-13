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


# class NodeEdgeFeatureEnhancer(nn.Module):
#     """
#     # This is the original code that works for stochastic training but throws errors for batch training
#     - Computes vᵢ⁰ defined in GAT-CADNet Paper Pg 4 equation 7
#     - Resultant of NodeEdgeFeatureEnhancer is of shape (num_nodes,128)
#     - Since we concatenate across dim=1, the output layer for node_mlp and edge_mlp should be int(output_dim/2)
#     """
#     def __init__(self, node_input_dim, edge_input_dim, output_dim):
#         super().__init__()
        
#         # Final output dimension
#         self.output_dim = output_dim

#         # MLP extracts node features and edge features
#         self.node_mlp = MLP(node_input_dim, int(output_dim/2))     # Map node features to 64-dimensional
#         self.edge_mlp = MLP(edge_input_dim, int(output_dim/2))     # Map edge features to 64-dimensional        

#     def forward(self, node_features, edge_features):
#         # Embed node features
#         node_emb = self.node_mlp(node_features)     # Node embedding v_hat_i

#         # Embed edge features
#         edge_emb = self.edge_mlp(edge_features)     # Edge embed e_hat_ij
        
#         print(f"Shapes of Embed node and Embed edge features: {node_emb.shape}, {edge_emb.shape}")

#         # Pool edge features of each node(max pooling)
#         if edge_emb.size(0) > 0:
#             pooled_edge_feats = torch.max(edge_emb, dim=0)[0].unsqueeze(0)
#         else:
#             pooled_edge_feats = torch.zeros(edge_emb.size(1), device=node_features.device).unsqueeze(0)  # Processing empty

#         # Spliced ​​node features and edge features after pooling
#         enhanced_node_features = torch.cat([node_emb, pooled_edge_feats], dim=1)

#         return enhanced_node_features

class NodeEdgeFeatureEnhancer(nn.Module):
    """
    - Computes vᵢ⁰ defined in GAT-CADNet Paper Pg 4 equation 7
    - Processes batches of graphs where each batch contains multiple combined graphs
    - Input node features shape: (total_nodes_in_batch, node_input_dim)
    - Input edge features shape: (total_edges_in_batch, edge_input_dim)
    - Output shape: (total_nodes_in_batch, output_dim)
    """
    def __init__(self, node_input_dim, edge_input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        
        # MLP for processing node and edge features
        self.node_mlp = MLP(node_input_dim, int(output_dim/2))
        self.edge_mlp = MLP(edge_input_dim, int(output_dim/2))

    def forward(self, node_features, edge_features, edge_index):
        """
        Args:
            node_features: Node features tensor (total_nodes_in_batch, node_input_dim)
            edge_features: Edge features tensor (total_edges_in_batch, edge_input_dim)
            edge_index: Edge connectivity (2, total_edges_in_batch)
        """
        # Verify that no nans in input
        if torch.isnan(node_features).any():
            print("NaN detected in node_features!")
        if torch.isnan(edge_features).any():
            print("NaN detected in edge_features!")
        if torch.isinf(node_features).any():
            print("Inf detected in node_features!")
        if torch.isinf(edge_features).any():
            print("Inf detected in edge_features!")

        
        
        # Process all node features in batch
        node_emb = self.node_mlp(node_features)  # Shape: (total_nodes_in_batch, output_dim/2)
        
        # Process all edge features in batch
        edge_emb = self.edge_mlp(edge_features)  # Shape: (total_edges_in_batch, output_dim/2)
        
        print(f"Shapes of Embed node and Embed edge features: {node_emb.shape}, {edge_emb.shape}")
        
        # Create a tensor to store aggregated edge features for each node
        num_nodes = node_features.size(0)
        aggregated_edge_features = torch.zeros(
            num_nodes, 
            self.output_dim // 2, 
            device=node_features.device
        )
        
        # Aggregate edge features for each node using edge_index
        source_nodes = edge_index[0]
        for node_idx in range(num_nodes):
            # Find all edges where this node is the source
            node_edges_mask = (source_nodes == node_idx)
            if node_edges_mask.any():
                # Get all edge features for this node and max pool them
                node_edge_features = edge_emb[node_edges_mask]
                node_aggregated = torch.max(node_edge_features, dim=0)[0]
                aggregated_edge_features[node_idx] = node_aggregated
        
        # Concatenate node embeddings with aggregated edge features
        enhanced_features = torch.cat(
            [node_emb, aggregated_edge_features], 
            dim=1
        )  # Shape: (total_nodes_in_batch, output_dim)
        
        return enhanced_features