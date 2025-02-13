"""
Changes made to a copy of batch_training.py that handles batch based training, attempts to optimise code
"""
import torch
from torch import optim, nn
from torch.utils.data import DataLoader # Using pytorch's DataLoader instead of torch_geometric.loader.DataLoader
from data.cad_dataset import CADDataset
from model.GATCADNet import GATCADNet
from module.NodeEdgeFeatureEnhancer import NodeEdgeFeatureEnhancer
from module.RSE import RSE
from tqdm import tqdm

def print_mem_stats():
    return(torch.cuda.memory_summary(device=None, abbreviated=True)) # provides short summary of memory statistics

# Custom collate function to merge a list of PyG Data objects into one disjoint graph.
def custom_collate(graph_list):
    """
    - To utilise GPU completely, we shift from stochastic to minibatch training.
    - Thus we concatenate nodes from multiple graphs to create a larger graph, so node_ids change based on when they are appended into this larger matrix and edge_ids also change correspondingly.
    - The previous adjacency matrices from various samples combine to become a "Block‐diagonal adjacency matrix"
    - This collate fn does all these steps
    
    """
    x_list, y_list, adj_list = [], [], []
    edge_attr_list, edge_index_list = [], []
    batch_vector = []
    cum_nodes = 0
    for i, data in enumerate(graph_list):
        n = data.x.size(0)
        x_list.append(data.x)
        y_list.append(data.y)
        adj_list.append(data.adj_matrix)
        edge_attr_list.append(data.edge_attr)
        # Shift edge indices by the cumulative number of nodes
        shifted_edge_index = data.edge_index + cum_nodes
        edge_index_list.append(shifted_edge_index)
        batch_vector.append(torch.full((n,), i, dtype=torch.long))
        cum_nodes += n
    # Stack node features and labels
    x_batch = torch.cat(x_list, dim=0)
    y_batch = torch.cat(y_list, dim=0)
    # Create a block-diagonal adjacency matrix for the batch
    adj_batch = torch.block_diag(*adj_list)
    edge_attr_batch = torch.cat(edge_attr_list, dim=0)
    edge_index_batch = torch.cat(edge_index_list, dim=1)
    batch_vector = torch.cat(batch_vector, dim=0)

    from torch_geometric.data import Data
    batched_data = Data(x=x_batch, y=y_batch, adj_matrix=adj_batch,
                        edge_index=edge_index_batch, edge_attr=edge_attr_batch,
                        batch_vec=batch_vector) #batch_vec is not a standard param of Data, it comes under kwargs
    batched_data.num_nodes = x_batch.size(0)
    return batched_data


if __name__ == "__main__":
    
    # 1. Detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"(1) Using device: {device}")
    
    # 2. Create dataset in PyG format
    dataset = CADDataset(svg_path='dataset/FloorplanCAD_sampledataset/verysmalltrainset')
    print(f"Dataset loaded with {len(dataset)} graphs.")

    # Create a mini-batch DataLoader using our custom collate function.
    batch_size = 2  # Number of graphs per mini-batch
    
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)  # shuffle=False for debugging purposes
    print(f"(2) Dataloader Created.....................") 
    
    # Training parameters
    n_heads = 8
    num_epochs = 3
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.99
    decay_rate = 0.7
    lr_decay_step = 20
    lambda_ins = 2
    
    # 3. Define loss functions.
    
    # BCEWithLogitsLoss = −(ylog(σ(x)) + (1−y)log(1−σ(x))) [y = groundtruth, x = logit]
    sem_loss = nn.CrossEntropyLoss() # Semantic Loss: CrossEntropyLoss
    ins_loss = nn.BCEWithLogitsLoss(reduction='none') # Instance loss: BinaryCrossEntropyLoss (numerically stable compared to applying torch.sigmoid() followed by nn.BCELoss())
    print("(3) Loss functions defined.......................")
    
    # Initialize model
    model = GATCADNet(
            in_channels=128,
            out_channels=128,
            num_heads=n_heads,
            num_stages=8,
        ).to(device)
    print("(4) Initialised and moved model to GPU.................")

    # Define the optimizer and learning rate scheduler.
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=decay_rate)
    print("(5) Optimizer and scheduler defined, entering training loop...")
    
    print_mem_stats()
    
    # Training loop (mini-batch style)
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        model.train()
        for batch in loader:
            # Move all batch tensors to device.
            batch.x = batch.x.to(device)
            batch.y = batch.y.to(device)
            batch.adj_matrix = batch.adj_matrix.to(device)
            batch.edge_index = batch.edge_index.to(device)
            batch.edge_attr = batch.edge_attr.to(device)
            batch.batch = batch.batch_vec.to(device)

            vertex_target = batch.y  # (total_nodes,)
            vertex_target_classes = int(vertex_target.max().item()) + 1
            adj_matrix_target = batch.adj_matrix.to(device)  # (total_nodes, total_nodes)
            num_nodes = batch.num_nodes

            # Enhance node features: for each node, use its own features and the features of its connected edges.
            vertex_features_list = []
            # (Optionally, instantiate the enhancer once outside the loop if it is stateless.)
            enhancer = NodeEdgeFeatureEnhancer(node_input_dim=7, edge_input_dim=7, output_dim=128).to(device)
            for node_id in range(batch.x.size(0)):
                node_features = batch.x[node_id].unsqueeze(0)  # (1, 7)
                # Find the indices where the node is the source in edge_index.
                edge_indices = (batch.edge_index[0] == node_id).nonzero(as_tuple=True)[0]
                edge_features = batch.edge_attr[edge_indices]
                node_new_features = enhancer(node_features, edge_features)  # (1, 128)
                vertex_features_list.append(node_new_features)
            # Stack enhanced node features into a tensor of shape (total_nodes, 128)
            vertex_features = torch.stack(vertex_features_list, dim=0).squeeze(1)

            # Process edge features and perform relative space encoding.
            # Create a tensor to hold edge features for every possible node pair.
            edge_feature_matrix = torch.zeros(num_nodes, num_nodes, 7, device=device)
            for i in range(batch.edge_index.size(1)):
                u, v = batch.edge_index[:, i]
                edge_feature_matrix[u, v] = batch.edge_attr[i]
                edge_feature_matrix[v, u] = batch.edge_attr[i]  # For undirected graphs.
            rse_module = RSE(in_channels=7, out_channels=n_heads).to(device)
            edge_feature_matrix = rse_module(edge_feature_matrix)  # Now shape: (num_nodes, num_nodes, n_heads)
            num_nodes = edge_feature_matrix.size(0)

            # Forward pass: get predicted node features and predicted adjacency matrix.
            predict_vertex_features, predict_adj_matrix = model(
                vertex_features=vertex_features,
                num_nodes=num_nodes,
                relative_encoding=edge_feature_matrix,
                num_classes=vertex_target_classes
            )

            # Compute semantic loss (for node classification).
            loss_sem = sem_loss(predict_vertex_features, vertex_target)

            """
            # Original code
            # Compute instance loss (for adjacency matrix prediction).
            # Build a weight matrix 'w' for the instance loss.
            w = torch.ones(num_nodes, num_nodes, device=device)
            for i in tqdm(range(num_nodes), desc="Computing weight matrix", leave=False):
                for j in range(num_nodes):
                    if vertex_target[i] == vertex_target[j] and adj_matrix_target[i, j] == 0:
                        w[i, j] = 20
                    elif vertex_target[i] == vertex_target[j] and adj_matrix_target[i, j] == 1:
                        w[i, j] = 2
                    elif vertex_target[i] != vertex_target[j] and adj_matrix_target[i, j] == 1:
                        w[i, j] = 0
            """
            
            print("================ Computing weight matrix 'w'========================")
            # Compute instance loss (for adjacency matrix prediction).
            # Build a weight matrix 'w' for the instance loss.
            w = torch.ones(num_nodes, num_nodes, device=device)
            # Create a mask for nodes in the same class vs. different classes
            mask_same_class = (vertex_target.unsqueeze(0) == vertex_target.unsqueeze(1))
            # Create a mask for edges that exist vs. do not exist
            mask_adj = (adj_matrix_target == 1)

            # 1. same class, no edge
            w[mask_same_class & (~mask_adj)] = 20

            # 2. same class, has edge
            w[mask_same_class & mask_adj] = 2

            # 3. different class, has edge
            w[~mask_same_class & mask_adj] = 0



            # Just printing where predict_adj_matrix and adj_matrix_target tensors are located for debugging purpose
            print(f"predict_adj_matrix is on: {predict_adj_matrix.device}, dtype: {predict_adj_matrix.dtype}, shape: {predict_adj_matrix.shape}")
            print(f"adj_matrix_target is on: {adj_matrix_target.device}, dtype: {adj_matrix_target.dtype}, shape: {adj_matrix_target.shape}")
            # Ensure predict_adj_matrix is on the same device as adj_matrix_target
            predict_adj_matrix = predict_adj_matrix.to(device)
            
            loss_ins = ins_loss(predict_adj_matrix, adj_matrix_target)
            weighted_loss_ins = loss_ins * w
            final_loss_ins = weighted_loss_ins.mean()

            total_loss = loss_sem + lambda_ins * final_loss_ins

            print(f"Batch Loss: {total_loss.item():.4f}, Semantic Loss: {loss_sem.item():.4f}, Instance Loss: {final_loss_ins.item():.4f}")

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # Step the scheduler every lr_decay_step epochs.
    if (epoch + 1) % lr_decay_step == 0:
        scheduler.step()

    print(f"After epoch {epoch+1}:")
    print_mem_stats()
    
    """
    Batchsize = 2
    Batch Loss: 10.2118, Semantic Loss: 3.5461, Instance Loss: 3.3328
    Batch Loss: 9.8833, Semantic Loss: 3.5473, Instance Loss: 3.1680
    Batch Loss: nan, Semantic Loss: nan, Instance Loss: 11.0005
    Batch Loss: nan, Semantic Loss: nan, Instance Loss: 3.0681
    Batch Loss: nan, Semantic Loss: nan, Instance Loss: 3.4139
    Batch Loss: nan, Semantic Loss: nan, Instance Loss: 7.6985
    Batch Loss: nan, Semantic Loss: nan, Instance Loss: 3.6722
    Batch Loss: nan, Semantic Loss: nan, Instance Loss: 4.6749
    """
    
    