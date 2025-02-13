import torch
import networkx as nx
from torch_geometric.data import Data
import numpy as np

"""
Data Objects:
torch_geometric.data.Data: A data object describing a homogeneous graph.(all nodes are of same type)
torch_geometric.data.HeteroData: A data object describing a heterogeneous graph.(nodes/edges are of multiple classes/types)

"""

# Convert networkx's graph to torch_geometric Data class
def convert_nx_to_tg_data(graph: nx.Graph) -> Data:
    """
    A function that converts a NetworkX graph (nx.Graph) into a PyTorch Geometric (torch_geometric.data.Data) object.
    
    - graph.nodes(data=True) returns an iterator of (node_id, node_attributes)
    - graph.edges(data=True) returns an iterator of (source, destination, edge_attributes)
    """
    # print(f"graph.nodes : {graph.nodes}") # List of node_ids
    # print(f"graph.edges : {len(graph.edges)}") # List of tuples, each tuple contains 2 node_ids that make the edge
    
    # Get the features of all points in the graph and splice them into vertex feature vectors
    node_features = [data['features'] for _, data in graph.nodes(data=True)]
    node_features = np.array(node_features).astype(np.float32) # (len(graph.nodes), 7)

    edge_features = [data['features'] for _, _, data in graph.edges(data=True)]
    edge_features = np.array(edge_features).astype(np.float32) # (len(graph.edges), 7)

    # Get the connection information of the edge(edge_index)
    edge_index = np.array(list(graph.edges)).T  # shape: (2, num_edges)

    # Get target, that is, the correct node classification
    target = [data.get('target') for _, data in graph.nodes(data=True)]

    # Convert data to pytorch tensor
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
    edge_features_tensor = torch.tensor(edge_features, dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    target_tensor = torch.tensor(target, dtype=torch.int64)

    # Create a Data object for torch_geometric
    data = Data(
        x=node_features_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_features_tensor,
        y=target_tensor
    )

    # Calculate the adjacency matrix
    adj_matrix = torch.zeros(data.num_nodes, data.num_nodes)
    
    for i in range(edge_index_tensor.size(1)):
        src, dst = edge_index_tensor[:, i]
        adj_matrix[src, dst] = 1
        adj_matrix[dst, src] = 1

    data.adj_matrix = adj_matrix 
    
    # print(f"\nCreated Data object:")
    # print(f"x: {data.x.shape}") # torch.Size([num_nodes, feature_dim]) node_feature_dim == 7 in GAT-CADNet
    # print(f"y: {data.y.shape}") # torch.Size([num_nodes])
    # print(f"edge_index: {data.edge_index.shape}") # torch.Size([2, num_edges]) "2" because edge is made of 2 nodes
    # print(f"edge_attr: {data.edge_attr.shape}") # torch.Size([num_edges, feature_dim]) edge_feature_dim == 7 in GAT-CADNet
    # print(f"adj_matrix: {type(data.adj_matrix)}") # torch.Size([num_nodes, num_nodes])

    return data
