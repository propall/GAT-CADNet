import torch
from torch import optim

from data.cad_dataset import CADDataset
from model.GATCADNet import GATCADNet
from module.NodeEdgeFeatureEnhancer import NodeEdgeFeatureEnhancer
from module.RSE import RSE
import torch.nn as nn

# Data set
dataset = CADDataset(svg_path='FloorplanCAD_sampledataset/train-00')

# Set training parameters
n_heads = 8
num_epochs = 100
lr = 0.001
beta1 = 0.9
beta2 = 0.99
decay_rate = 0.7
lr_decay_step = 20
lambda_ins = 2

# Loss function
# Semantic Loss: CrossEntropyLoss
sem_loss = nn.CrossEntropyLoss()

# Instance loss: BinaryCrossEntropyLoss
ins_loss = nn.BCEWithLogitsLoss(reduction='none')  # 使用none，以便我们可以手动加权

# Model
model = GATCADNet(
        in_channels=128,
        out_channels=128,
        num_heads=n_heads,
        num_stages=8,
    )

# Adam Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=decay_rate)

# Model training process
for epoch in range(num_epochs):
    model.train()
    for graph in dataset:
        vertex_target = graph.y  # (num_nodes, 1)
        vertex_target_classes = int(vertex_target.max()) + 1
        adj_matrix_target = graph.adj_matrix    # Real adjacency matrix
        num_nodes = graph.num_nodes
        vertex_features_list = []

        """ Use edge features to enhance node features """
        for node_id in range(graph.x.size(0)):  # Iterate through all nodes
            # (1, 7)
            node_features = graph.x[node_id].unsqueeze(0)  # Get the characteristics of the current node

            # Get the index of the edges connected to the current node
            edge_indices = (graph.edge_index[0] == node_id).nonzero(as_tuple=True)[0]

            # Get the features of these edges
            # (num_edges（Number of neighbors）, 7)
            edge_features = graph.edge_attr[edge_indices]
            enhancer = NodeEdgeFeatureEnhancer(node_input_dim=7, edge_input_dim=7, output_dim=128)
            # (1, 128)
            node_new_features = enhancer(node_features, edge_features)
            vertex_features_list.append(node_new_features)

        # Use torch.stack to splice the tensors in the list into a (num_nodes, 128) tensor
        vertex_features = torch.stack(vertex_features_list, dim=0).squeeze(1)

        """ Process edge features and perform relative space encoding """
        # Create a tensor with shape (num_nodes, num_nodes, 7) to store the features of all edges
        edge_feature_matrix = torch.zeros(num_nodes, num_nodes, 7)
        RSE = RSE(in_channels=7, out_channels=n_heads)

        # Fill the features of each edge into edge_feature_matrix
        for i in range(graph.edge_index.shape[1]):
            u, v = graph.edge_index[:, i]  # Get the node pair of edges (u, v)
            edge_feature_matrix[u, v] = graph.edge_attr[i]  # Fill edge features to (u, v) position
            edge_feature_matrix[v, u] = graph.edge_attr[i]  # For undirected graphs, fill (v, u) positions
        # Processing this tensor using RSE (num_nodes, num_nodes, 7) -> (num_nodes, num_nodes, n_heads)
        edge_feature_matrix = RSE(edge_feature_matrix)
        num_nodes = edge_feature_matrix.size(0)

        """ Training GATCADNet using data """
        # Node features and adjacency matrix predicted using model
        # predict_vertex_features: (num_nodes, 30)  predict_adj_matrix: (num_nodes, num_nodes)
        predict_vertex_features, predict_adj_matrix = model(
            vertex_features=vertex_features,
            num_nodes=num_nodes,
            relative_encoding=edge_feature_matrix,
            num_classes=vertex_target_classes
        )

        # Semantic Loss
        loss_sem = sem_loss(predict_vertex_features, vertex_target)

        # Instance loss
        # Constructing a weight matrix
        w = torch.ones(num_nodes, num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if vertex_target[i] == vertex_target[j] and adj_matrix_target[i, j] == 0:
                    w[i, j] = 20
                elif vertex_target[i] == vertex_target[j] and adj_matrix_target[i, j] == 1:
                    w[i, j] = 2
                elif vertex_target[i] != vertex_target[j] and adj_matrix_target[i, j] == 1:
                    w[i, j] = 0

        loss_ins = ins_loss(predict_adj_matrix, adj_matrix_target)

        weighted_loss_ins = loss_ins * w

        final_loss_ins = weighted_loss_ins.mean()

        # Calculate total loss
        total_loss = loss_sem + lambda_ins * final_loss_ins

        # Output loss
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {total_loss.item():.4f}, "
              f"Semantic Loss: {loss_sem.item():.4f}, "
              f"Instance Loss: {final_loss_ins.item():.4f}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Adjust learning rate
    if epoch % 20 == 0:
        scheduler.step()
