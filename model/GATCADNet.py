"""
GATCADNet: Paper Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.CEE import CEE
from module.GATLayer import GATLayer


class GATCADNetMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class GATCADNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_stages):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param num_heads: Number of attention heads
        :param num_stages: Number of phases of the network
        :param relative_encoding: Relative position encoding
        :param num_classes: Total number of points categories
        """
        super().__init__()
        self.out_channels = out_channels

        self.num_stages = num_stages
        self.num_heads = num_heads

        self.gat_layers = nn.ModuleList([
            GATLayer(in_channels, out_channels, num_heads) for _ in range(num_stages)
        ])

    def forward(self, vertex_features, num_nodes, relative_encoding, num_classes):
        """
        :param num_classes: Total number of points categories
        :param relative_encoding: Relative position encoding
        :param num_nodes: Number of nodes
        :param vertex_features: Input node characteristics
        :return: Classification results
        """

        mlp_module = GATCADNetMLP(self.out_channels, num_classes).to(vertex_features.device)
        # Cascade edge encoding for CEE module
        cascaded_edge_encoding = torch.zeros(num_nodes, num_nodes, self.num_heads, device=vertex_features.device)

        for s in range(self.num_stages):
            vertex_features, attention_scores = self.gat_layers[s](vertex_features, relative_encoding, num_nodes)
            cascaded_edge_encoding += attention_scores

        cee_module = CEE(in_channels=256+8, out_channels=128, num_nodes=num_nodes)
        predict_adj_matrix = cee_module(cascaded_edge_encoding, vertex_features)

        # (num_nodes, num_classes)
        predict_vertex_features = mlp_module(vertex_features)
        predict_vertex_features = F.softmax(predict_vertex_features, dim=1)
        
        # print(f"predict_vertex_features, predict_adj_matrix: {predict_vertex_features.shape}, {predict_adj_matrix.shape}")

        return predict_vertex_features, predict_adj_matrix










