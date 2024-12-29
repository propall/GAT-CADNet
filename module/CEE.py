"""
CEE module: CEE 模块
"""

import torch
import torch.nn as nn


class CEE(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()

        self.num_nodes = num_nodes

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )

    def forward(self, cascaded_edge_encoding, vertex_features):
        """
        :param cascaded_edge_encoding: 级联边编码(num_nodes, num_nodes, n_heads)
        :param vertex_features: 经过GAT的节点特征(num_nodes, 128)
        :return: 预测的邻接矩阵(num_nodes, num_nodes)
        """

        # edge_features = torch.zeros((self.num_nodes, self.num_nodes, (256+8)), dtype=torch.float32)
        #
        # for i in range(self.num_nodes):
        #     for j in range(self.num_nodes):
        #         c_ij = cascaded_edge_encoding[i][j]
        #         v_i = vertex_features[i]
        #         v_j = vertex_features[j]
        #         edge_features[i, j] = torch.cat([c_ij, v_i, v_j], dim=0)

        # out = self.mlp(edge_features)

        # 消耗内存过大，使用随机张量进行模拟
        edge_features_random = torch.randn(1101, 1101, 264)

        out = self.mlp(edge_features_random)

        out = torch.sigmoid(out)

        out = out.view(1101, 1101)

        return out
