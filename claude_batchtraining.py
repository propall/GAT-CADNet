import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from data.cad_dataset import CADDataset
from model.GATCADNet import GATCADNet
from module.NodeEdgeFeatureEnhancer import NodeEdgeFeatureEnhancer
from module.RSE import RSE
from tqdm import tqdm
import time
import logging
from typing import Dict, Tuple
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OptimizedGATCADTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.99,
        lambda_ins: float = 2.0
    ):
        self.model = model
        self.device = device
        self.lambda_ins = lambda_ins
        
        # Initialize optimizer and losses
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
        self.scaler = GradScaler()  # For mixed precision training
        self.sem_loss = nn.CrossEntropyLoss() # Semantic loss
        self.ins_loss = nn.BCEWithLogitsLoss(reduction='none') # Instance loss: BCEWithLogitsLoss = −(ylog(σ(x)) + (1−y)log(1−σ(x))) [y = groundtruth, x = logit] (numerically stable compared to applying torch.sigmoid() followed by nn.BCELoss())
        
        # Initialize enhancer and RSE module
        self.node_enhancer = NodeEdgeFeatureEnhancer(
            node_input_dim=7,
            edge_input_dim=7,
            output_dim=128
        ).to(device) # Compute vᵢ⁰ defined in GAT-CADNet Paper Pg 4 equation 7
        self.rse_module = RSE(in_channels=7, out_channels=8).to(device)
        
    @torch.no_grad()
    def compute_weight_matrix(self, vertex_target: torch.Tensor, adj_matrix_target: torch.Tensor) -> torch.Tensor:
        """Vectorized computation of weight matrix
        Source: GAT-CADNet paper Pg5 weights table located below equation 17 
        """
        # Create comparison masks
        node_i = vertex_target.unsqueeze(1)
        node_j = vertex_target.unsqueeze(0)
        same_class_mask = (node_i == node_j)
        no_edge_mask = (adj_matrix_target == 0)
        has_edge_mask = (adj_matrix_target == 1)
        
        # Initialize and compute weights vectorized
        w = torch.ones_like(adj_matrix_target, dtype=torch.float, device=self.device)
        w = torch.where(same_class_mask & no_edge_mask, torch.tensor(20., device=self.device), w)
        w = torch.where(same_class_mask & has_edge_mask, torch.tensor(2., device=self.device), w)
        w = torch.where(~same_class_mask & has_edge_mask, torch.tensor(0., device=self.device), w)
        
        if torch.isnan(w).any():
            print("Warning: NaN detected in weight matrix w!")
            nan_count = torch.isnan(w).sum().item()
            print(f"Number of NaNs in w: {nan_count}")

        
        return w

    # def enhance_node_features_batch(
    #     self,
    #     x: torch.Tensor,
    #     edge_index: torch.Tensor,
    #     edge_attr: torch.Tensor
    # ) -> torch.Tensor:
    #     """Vectorized node feature enhancement for entire batch"""
    #     # Sort edges by source nodes
    #     node_indices = edge_index[0]
    #     sorted_indices = torch.argsort(node_indices)
    #     sorted_edges = edge_index[:, sorted_indices]
    #     sorted_edge_attr = edge_attr[sorted_indices]
        
    #     # Find unique nodes and their edge counts
    #     unique_nodes, counts = torch.unique(sorted_edges[0], return_counts=True)
    #     max_edges = counts.max()
        
    #     # Create padded edge features tensor
    #     padded_edge_features = torch.zeros(
    #         x.size(0), max_edges, edge_attr.size(1),
    #         device=self.device
    #     )
        
    #     # Fill in actual edge features
    #     current_idx = 0
    #     for node_idx, count in enumerate(counts):
    #         padded_edge_features[unique_nodes[node_idx], :count] = sorted_edge_attr[current_idx:current_idx + count]
    #         current_idx += count
        
    #     # Process through enhancer in one go
    #     return self.node_enhancer(x.unsqueeze(1), padded_edge_features).squeeze(1)

    # def enhance_node_features_batch(
    #     self,
    #     x: torch.Tensor,
    #     edge_index: torch.Tensor,
    #     edge_attr: torch.Tensor
    # ) -> torch.Tensor:
    #     """
    #     Vectorized node feature enhancement for entire batch
    #     Implements equation 7 from GAT-CADNet paper for initial node features
    #     """

    #     batch_size = x.size(0)
    #     enhanced_features_list = []       
    #     # Process each node separately to maintain proper dimensionality
    #     for i in range(batch_size):
    #         # Find all edges connected to this node
    #         edges_mask = (edge_index[0] == i)
    #         node_edges = edge_attr[edges_mask]          
    #         if node_edges.size(0) == 0:
    #             # If no edges, use zero tensor with proper shape
    #             node_edges = torch.zeros(1, edge_attr.size(1), device=self.device)            
    #         # Process through enhancer
    #         # x[i] shape: [7], needs to be [1, 7]
    #         # node_edges shape: [num_edges, 7]
    #         node_feature = x[i].unsqueeze(0)  # Shape: [1, 7]
    #         enhanced = self.node_enhancer(node_feature, node_edges.unsqueeze(0))  # Add batch dimension
    #         enhanced_features_list.append(enhanced.squeeze(0))        
    #     # Stack all enhanced features
    #     return torch.stack(enhanced_features_list)
    
    def enhance_node_features_batch(self, x, edge_index, edge_attr):
        """
        Process a batch of graphs that have been combined into a single large graph
        
        Args:
            x: Node features tensor (total_nodes_in_batch, node_input_dim)
            edge_index: Edge connectivity (2, total_edges_in_batch)
            edge_attr: Edge features tensor (total_edges_in_batch, edge_input_dim)
        Returns:
            Enhanced node features (total_nodes_in_batch, output_dim)
        """
        return self.node_enhancer(x, edge_attr, edge_index)




    def compute_edge_features(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Compute edge feature matrix efficiently"""
        edge_feature_matrix = torch.zeros(
            num_nodes, num_nodes, edge_attr.size(1),
            device=self.device
        )
        edge_feature_matrix[edge_index[0], edge_index[1]] = edge_attr
        edge_feature_matrix[edge_index[1], edge_index[0]] = edge_attr  # Symmetric
        return self.rse_module(edge_feature_matrix)

    def train_batch(self, batch) -> Tuple[float, float, float]:
        """Process a single training batch"""
        # Move batch to device (if not already done)
        batch = batch.to(self.device)
        
        with autocast():  # Enable mixed precision
            # Enhanced feature computation (vectorized)
            vertex_features = self.enhance_node_features_batch(
                batch.x, batch.edge_index, batch.edge_attr
            )
            
            # Edge feature processing (vectorized)
            edge_features = self.compute_edge_features(
                batch.edge_index, batch.edge_attr, batch.num_nodes
            )
            
            # Forward pass
            pred_vertex_features, pred_adj_matrix = self.model(
                vertex_features=vertex_features,
                num_nodes=batch.num_nodes,
                relative_encoding=edge_features,
                num_classes=int(batch.y.max().item()) + 1
            )
            
            # Loss computation
            loss_sem = self.sem_loss(pred_vertex_features, batch.y)
            w = self.compute_weight_matrix(batch.y, batch.adj_matrix)
            
            # # Just printing where predict_adj_matrix and adj_matrix_target tensors are located for debugging purpose
            # print(f"predict_adj_matrix is on: {pred_adj_matrix.device}, dtype: {pred_adj_matrix.dtype}, shape: {pred_adj_matrix.shape}")
            # print(f"batch.adj_matrix is on: {batch.adj_matrix.device}, dtype: {batch.adj_matrix.dtype}, shape: {batch.adj_matrix.shape}")
            
            # Ensure predict_adj_matrix is on the same device as adj_matrix_target
            pred_adj_matrix = pred_adj_matrix.to(self.device)
            
            loss_ins = self.ins_loss(pred_adj_matrix, batch.adj_matrix)
            weighted_loss_ins = (loss_ins * w).mean()
            
            # Calculate panoptic loss (Source: GAT-CADNet paper Pg 5, equation 18)
            total_loss = loss_sem + self.lambda_ins * weighted_loss_ins 
        
        # Optimization step with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return total_loss.item(), loss_sem.item(), weighted_loss_ins.item()

def custom_collate(batch):
    """
    Optimized collate function for creating mini-batches
    
    - To utilise GPU completely, we shift from stochastic to minibatch training.
    - Thus we concatenate nodes from multiple graphs to create a larger graph, so node_ids change based on when they are appended into this larger matrix and edge_ids also change correspondingly.
    - The previous adjacency matrices from various samples combine to become a "Block‐diagonal adjacency matrix"
    - This collate fn does all these steps
    
    """
    x_list, y_list, adj_list = [], [], []
    edge_attr_list, edge_index_list = [], []
    batch_vector = []
    cum_nodes = 0
    
    # Process all samples in batch at once
    for i, data in enumerate(batch):
        n = data.x.size(0)
        x_list.append(data.x)
        y_list.append(data.y)
        adj_list.append(data.adj_matrix)
        edge_attr_list.append(data.edge_attr)
        # Shift edge indices by the cumulative number of nodes
        edge_index_list.append(data.edge_index + cum_nodes)
        batch_vector.append(torch.full((n,), i, dtype=torch.long))
        cum_nodes += n
    
    # Concatenate all tensors at once
    x_batch = torch.cat(x_list, dim=0)
    y_batch = torch.cat(y_list, dim=0)
    # Create a block-diagonal adjacency matrix for the batch
    adj_batch = torch.block_diag(*adj_list)
    edge_attr_batch = torch.cat(edge_attr_list, dim=0)
    edge_index_batch = torch.cat(edge_index_list, dim=1)
    batch_vector = torch.cat(batch_vector, dim=0)
    
    from torch_geometric.data import Data
    batched_data = Data(
        x=x_batch,
        y=y_batch,
        adj_matrix=adj_batch,
        edge_index=edge_index_batch,
        edge_attr=edge_attr_batch,
        batch_vec=batch_vector, # batch_vec is not a standard param of Data, it comes under kwargs
        num_nodes=x_batch.size(0)
    )
    
    return batched_data

def train(
    dataset_path: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    n_heads: int = 2,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.99,
    decay_rate: float = 0.7,
    lr_decay_step: int = 20,
    lambda_ins: float = 2,
    num_workers: int = 4
) -> Dict:
    """
    Main training function
    """
    # Set up device and logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = CADDataset(svg_path=dataset_path)
    logging.info(f"Dataset loaded with {len(dataset)} graphs")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    # Initialize model and trainer
    model = GATCADNet(
        in_channels=128,
        out_channels=128,
        num_heads=n_heads,
        num_stages=8
    ).to(device)
    ####
    
    trainer = OptimizedGATCADTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        lambda_ins=lambda_ins
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        trainer.optimizer,
        step_size=lr_decay_step,
        gamma=decay_rate
    )
    
    # Training loop
    metrics = {
        'epoch_losses': [],
        'batch_losses': [],
        'training_time': 0
    }
    
    start_time = time.time()
    
    try:
        for epoch in range(num_epochs):
            epoch_loss = 0
            batch_count = 0
            
            # Progress bar for batches
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                total_loss, sem_loss, ins_loss = trainer.train_batch(batch)
                epoch_loss += total_loss
                batch_count += 1
                
                metrics['batch_losses'].append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'total_loss': total_loss,
                    'semantic_loss': sem_loss,
                    'instance_loss': ins_loss
                })
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss:.4f}',
                    'sem_loss': f'{sem_loss:.4f}',
                    'ins_loss': f'{ins_loss:.4f}'
                })
                
                # Periodic memory cleanup
                if (batch_idx + 1) % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Step the scheduler
            if (epoch + 1) % lr_decay_step == 0:
                scheduler.step()
            
            # Record epoch metrics
            avg_epoch_loss = epoch_loss / batch_count
            metrics['epoch_losses'].append(avg_epoch_loss)
            
            logging.info(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
            
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    
    metrics['training_time'] = time.time() - start_time
    logging.info(f"Training completed in {metrics['training_time']:.2f} seconds")
    
    return metrics

if __name__ == "__main__":
    # Training configuration
    config = {
        'dataset_path': 'dataset/FloorplanCAD_sampledataset/verysmalltrainset',
        'num_epochs': 2,
        'batch_size': 3,
        'n_heads': 8,
        'learning_rate': 0.001,
        'beta1': 0.9,
        'beta2': 0.99,
        'decay_rate': 0.7,
        'lr_decay_step': 20,
        'lambda_ins': 2,
        'num_workers': 4
    }
    
    
    # Start training
    metrics = train(**config)
    
    # Log final metrics
    logging.info("Final Training Metrics:")
    logging.info(f"Total training time: {metrics['training_time']:.2f} seconds")
    logging.info(f"Final average loss: {metrics['epoch_losses'][-1]:.4f}")