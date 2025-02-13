"""
Changes made to a copy of main.py that handles batch based training
"""
import torch
from torch import optim
from data.cad_dataset import CADDataset
from model.GATCADNet import GATCADNet
from module.NodeEdgeFeatureEnhancer import NodeEdgeFeatureEnhancer
from module.RSE import RSE
import torch.nn as nn

from tqdm import tqdm

from torch_geometric.loader import DataLoader

# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create graph in PyG format and combine them to create a dataloader with 16 samples as batch
dataset = CADDataset(svg_path='dataset/FloorplanCAD_sampledataset/train-00')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Batching 16 graphs/SVGs at a time


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
ins_loss = nn.BCEWithLogitsLoss(reduction='none')  # Use none so we can manually weight
# BCEWithLogitsLoss = −(ylog(σ(x)) + (1−y)log(1−σ(x))) [y = groundtruth, x = logit]
print("Defined loss fns.................")


# Initialize model
model = GATCADNet(
        in_channels=128,
        out_channels=128,
        num_heads=n_heads,
        num_stages=8,
    ).to(device)
print("Initialised and moved model to GPU.................")

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=decay_rate)

print("Defined optimizer and scheduler, entering training loop.................")

print("############### Mem Stats before training starts: ###############")
print(torch.cuda.memory_summary(device=None, abbreviated=True)) # provides short summary of memory statistics

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}] - Training Started")
    model.train() # Set model to training mode
    
    for batch in data_loader:
        batch = batch.to(device)  # Move entire batch to GPU
        
        # Extract batched data
        # Node and adjacency matrix targets
        vertex_target = batch.y # tensor containing the semantic class labels for each node.
        batch_size = batch.num_graphs  # Number of graphs in batch
        vertex_target_classes = int(vertex_target.max()) + 1
        adj_matrix_target = batch.adj_matrix # Real adjacency matrix
        num_nodes = batch.x.shape[0]
        print(f"vertex_target, vertex_target_classes, num_nodes, adj_matrix_target: {vertex_target.shape}, {vertex_target_classes}, {num_nodes}, {adj_matrix_target.shape}") # torch.Size([1079]), 30, 1079, torch.Size([1079, 1079])
        
        """ Process Node Features with Edge Features """
        vertex_features_list = []
        enhancer = NodeEdgeFeatureEnhancer(node_input_dim=7, edge_input_dim=7, output_dim=128).to(device)

        """ Use edge features to enhance node features """
        for node_id in range(batch.x.size(0)):  # Iterate through all nodes in batch
            node_features = batch.x[node_id].unsqueeze(0)  # Get the characteristics of the current node
            edge_indices = (batch.edge_index[0] == node_id).nonzero(as_tuple=True)[0]
            edge_features = batch.edge_attr[edge_indices]
            node_new_features = enhancer(node_features, edge_features)
            vertex_features_list.append(node_new_features)

        vertex_features = torch.stack(vertex_features_list, dim=0).squeeze(1)
        print(f"Shape of vertex_features: {vertex_features}")
        
        
        """ Process Edge Features and Apply Relative Spatial Encoding (RSE) """
        edge_feature_matrix = torch.zeros(num_nodes, num_nodes, 7, device=device)
        rse_module = RSE(in_channels=7, out_channels=n_heads).to(device) # Original https://github.com/Liberation-happy/GAT-CADNet code is RSE = RSE(in_channels=7, out_channels=n_heads), since both the classname and instance object have the same names, there is a naming conflict and thus uses the class instead of this computed value later, throwing an error

        # Fill the features of each edge into edge_feature_matrix
        for i in range(batch.edge_index.shape[1]):
            u, v = batch.edge_index[:, i]  # Get the node pair of edges (u, v)
            edge_feature_matrix[u, v] = batch.edge_attr[i]  # Fill edge features to (u, v) position
            edge_feature_matrix[v, u] = batch.edge_attr[i]  # For undirected graphs, fill (v, u) positions
            
        edge_feature_matrix = rse_module(edge_feature_matrix)

        """ Forward Pass Through GATCADNet """
        predict_vertex_features, predict_adj_matrix = model(
            vertex_features=vertex_features,
            num_nodes=num_nodes,
            relative_encoding=edge_feature_matrix,
            num_classes=vertex_target_classes
        )
        
        """ Compute Losses """
        # 1) Semantic Loss
        loss_sem = sem_loss(predict_vertex_features, batch.y)
        print(f"loss_sem: {loss_sem}")

        # Constructing batch-wise weight matrix
        w = torch.ones(num_nodes, num_nodes, device=device) # This constructs weight matrix on GPU while "w = torch.ones(num_nodes, num_nodes).to(device)" first creates w on CPU and then moves to GPU
        for i in tqdm(range(num_nodes), desc="Rows"):
            for j in range(num_nodes):
                if vertex_target[i] == vertex_target[j] and adj_matrix_target[i, j] == 0:
                    w[i, j] = 20
                elif vertex_target[i] == vertex_target[j] and adj_matrix_target[i, j] == 1:
                    w[i, j] = 2
                elif vertex_target[i] != vertex_target[j] and adj_matrix_target[i, j] == 1:
                    w[i, j] = 0
        
        print("After computing weight matrix:")
        print(torch.cuda.memory_summary(device=None, abbreviated=True)) # provides provides short summary of memory statistics of memory statistics

        # Just printing where predict_adj_matrix and adj_matrix_target tensors are located for debugging purpose
        print(f"predict_adj_matrix is on: {predict_adj_matrix.device}, dtype: {predict_adj_matrix.dtype}, shape: {predict_adj_matrix.shape}") # cpu, dtype: torch.float32, shape: torch.Size([371, 371])
        print(f"adj_matrix_target is on: {adj_matrix_target.device}, dtype: {adj_matrix_target.dtype}, shape: {adj_matrix_target.shape}") # cuda:0, dtype: torch.float32, shape: torch.Size([371, 371])

        # 2) Instance Loss
        # Ensure predict_adj_matrix is on the same device as adj_matrix_target
        predict_adj_matrix = predict_adj_matrix.to(device)
        # print(f"Shapes of predict_adj_matrix, adj_matrix_target: {predict_adj_matrix.shape}, {adj_matrix_target.shape}") # torch.Size([1101, 1101]), torch.Size([1079, 1079])
        loss_ins = ins_loss(predict_adj_matrix, adj_matrix_target)

        weighted_loss_ins = loss_ins * w
        final_loss_ins = weighted_loss_ins.mean()

        # Total loss
        total_loss = loss_sem + lambda_ins * final_loss_ins

        # Print loss metrics
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {total_loss.item():.4f}, "
              f"Semantic Loss: {loss_sem.item():.4f}, "
              f"Instance Loss: {final_loss_ins.item():.4f}")

        # Backpropagation and optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Adjust learning rate
    if epoch % 20 == 0:
        scheduler.step()
        
print("Training Complete!")