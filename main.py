import torch
from torch import optim

from data.cad_dataset import CADDataset
from model.GATCADNet import GATCADNet
from module.NodeEdgeFeatureEnhancer import NodeEdgeFeatureEnhancer
from module.RSE import RSE
import torch.nn as nn

from tqdm import tqdm

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data set
dataset = CADDataset(svg_path='dataset/FloorplanCAD_sampledataset/train-00')

print("Loaded Dataset through dataloader.................")

# Set training parameters
n_heads = 8
num_epochs = 3
lr = 0.001
beta1 = 0.9
beta2 = 0.99
decay_rate = 0.7
lr_decay_step = 20
lambda_ins = 2

# Loss function
# Semantic Loss: CrossEntropyLoss
sem_loss = nn.CrossEntropyLoss()

# Instance loss: BinaryCrossEntropyLoss (numerically stable compared to applying torch.sigmoid() followed by nn.BCELoss())
# BCEWithLogitsLoss = −(ylog(σ(x)) + (1−y)log(1−σ(x))) [y = groundtruth, x = logit]
ins_loss = nn.BCEWithLogitsLoss(reduction='none')  # Use none so we can manually weight

print("Defined loss fns.................")
# Model
model = GATCADNet(
        in_channels=128,
        out_channels=128,
        num_heads=n_heads,
        num_stages=8,
    ).to(device)
print("Initialised and moved model to GPU.................")

# Adam Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=decay_rate)
print("Defined optimizer and scheduler, entering training loop.................")

print("Before training starts:")
print(torch.cuda.memory_summary(device=None, abbreviated=True)) # provides short summary of memory statistics

# Model training process
for epoch in range(num_epochs):
    print("Set model to train mode and start new epoch.................")
    model.train()
    for graph in dataset:
        print("Parsing through dataset...........")
        vertex_target = graph.y.to(device)  # (num_nodes, 1) # graph.y is a tensor containing the semantic class labels for each node.
        
        # Also move graph.x, graph.adj_matrix, graph.edge_attr, graph.edge_index to device:
        graph.x = graph.x.to(device)
        graph.adj_matrix = graph.adj_matrix.to(device)
        graph.edge_attr = graph.edge_attr.to(device)
        graph.edge_index = graph.edge_index.to(device)
        
        vertex_target_classes = int(vertex_target.max()) + 1
        adj_matrix_target = graph.adj_matrix.to(device)    # Real adjacency matrix
        num_nodes = graph.num_nodes
        
        # print(f"vertex_target, vertex_target_classes, num_nodes, adj_matrix_target: {vertex_target.shape}, {vertex_target_classes}, {num_nodes}, {adj_matrix_target.shape}") # torch.Size([1079]), 30, 1079, torch.Size([1079, 1079])
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
            enhancer = NodeEdgeFeatureEnhancer(node_input_dim=7, edge_input_dim=7, output_dim=128).to(device)
            # (1, 128)
            node_new_features = enhancer(node_features, edge_features)
            vertex_features_list.append(node_new_features)

        # Use torch.stack to splice the tensors in the list into a (num_nodes, 128) tensor
        vertex_features = torch.stack(vertex_features_list, dim=0).squeeze(1)

        """ Process edge features and perform relative space encoding """
        # Create a tensor with shape (num_nodes, num_nodes, 7) to store the features of all edges
        edge_feature_matrix = torch.zeros(num_nodes, num_nodes, 7, device=device)
        rse_module = RSE(in_channels=7, out_channels=n_heads).to(device) # Original https://github.com/Liberation-happy/GAT-CADNet code is RSE = RSE(in_channels=7, out_channels=n_heads), since both the classname and instance object have the same names, there is a naming conflict and thus uses the class instead of this computed value later, throwing an error

        # Fill the features of each edge into edge_feature_matrix
        for i in range(graph.edge_index.shape[1]):
            u, v = graph.edge_index[:, i]  # Get the node pair of edges (u, v)
            edge_feature_matrix[u, v] = graph.edge_attr[i]  # Fill edge features to (u, v) position
            edge_feature_matrix[v, u] = graph.edge_attr[i]  # For undirected graphs, fill (v, u) positions
        # Processing this tensor using RSE (num_nodes, num_nodes, 7) -> (num_nodes, num_nodes, n_heads)
        edge_feature_matrix = rse_module(edge_feature_matrix)
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
        print(f"loss_sem is on: {loss_sem.device}, dtype: {loss_sem.dtype}, shape: {loss_sem.shape}") # cuda:0, dtype: torch.float32, shape: torch.Size([])
        print(f"loss_sem: {loss_sem}")

        # Instance loss
        print("Constructing a weight matrix")
        # Constructing a weight matrix
        w = torch.ones(num_nodes, num_nodes, device=device) # This constructs weight matrix on GPU while "w = torch.ones(num_nodes, num_nodes).to(device)" first creates w on CPU and then moves to GPU
        for i in tqdm(range(num_nodes), desc="Rows"):
            for j in range(num_nodes):
                if vertex_target[i] == vertex_target[j] and adj_matrix_target[i, j] == 0:
                    w[i, j] = 20
                elif vertex_target[i] == vertex_target[j] and adj_matrix_target[i, j] == 1:
                    w[i, j] = 2
                elif vertex_target[i] != vertex_target[j] and adj_matrix_target[i, j] == 1:
                    w[i, j] = 0
        
        print("Computed weight matrix")
        
        print("After computing weight matrix:")
        print(torch.cuda.memory_summary(device=None, abbreviated=True)) # provides provides short summary of memory statistics of memory statistics

        # Just printing where predict_adj_matrix and adj_matrix_target tensors are located for debugging purpose
        print(f"predict_adj_matrix is on: {predict_adj_matrix.device}, dtype: {predict_adj_matrix.dtype}, shape: {predict_adj_matrix.shape}") # cpu, dtype: torch.float32, shape: torch.Size([371, 371])
        print(f"adj_matrix_target is on: {adj_matrix_target.device}, dtype: {adj_matrix_target.dtype}, shape: {adj_matrix_target.shape}") # cuda:0, dtype: torch.float32, shape: torch.Size([371, 371])

        # Ensure predict_adj_matrix is on the same device as adj_matrix_target
        predict_adj_matrix = predict_adj_matrix.to(device)

        # print(f"Shapes of predict_adj_matrix, adj_matrix_target: {predict_adj_matrix.shape}, {adj_matrix_target.shape}") # torch.Size([1101, 1101]), torch.Size([1079, 1079])
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
