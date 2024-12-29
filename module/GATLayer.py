"""
GATLayer: 图注意力网络层
对节点特征进行处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        """
        :param in_channels: 输入特征的通道数
        :param out_channels: 输出特征的通道数
        :param num_heads: 注意力头的数量
        """
        super().__init__()

        self.num_heads = num_heads
        self.out_channels = out_channels
        self.d = out_channels // num_heads

        # 线性变换层，用于Q, K, V的生成 (num_nodes, d) d = out_channels / num_heads（参见Transformer）
        self.w_q = nn.Linear(in_channels, out_channels)
        self.w_k = nn.Linear(in_channels, out_channels)
        self.w_v = nn.Linear(in_channels, out_channels)

        # MLP模块，用于最终的特征变换
        self.mlp = nn.Sequential(
            nn.Linear(self.d, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, vertex_features, relative_encoding, num_nodes):
        """
        :param num_nodes: 节点数量
        :param relative_encoding: 相对位置编码，来自RSE模块
        :param vertex_features: 输入的节点特征，大小为[num_nodes, D_in], 其中num_nodes是节点数，D_in是输入特征通道数
        :return:
        """

        # 投影到Q, K, V矩阵 (num_nodes, d, num_heads)
        q = self.w_q(vertex_features).view(num_nodes, self.d, self.num_heads)
        k = self.w_k(vertex_features).view(num_nodes, self.d, self.num_heads)
        v = self.w_v(vertex_features).view(num_nodes, self.d, self.num_heads)

        k_transposed = k.permute(2, 1, 0)
        q = q.permute(2, 0, 1)

        # 计算attention_scores
        attention_scores = torch.matmul(q, k_transposed).permute(1, 2, 0)

        # 添加相对位置编码
        attention_scores = attention_scores + relative_encoding

        # Softmax在每一行计算权重
        attention_scores = F.softmax(attention_scores, dim=-1)

        # 计算加权值v (num_nodes, num_heads, out_channels)
        v_transposed = v.permute(0, 2, 1)
        attention_output = torch.matmul(attention_scores, v_transposed).sum(dim=1)

        # 合并所有头的输出(num_nodes, num_heads * out_channels)
        attention_output = attention_output.view(num_nodes, -1)

        output = self.mlp(attention_output)

        return output, attention_scores




