"""
GATLayer: Graph Attention Network Layer
Process node features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        """
        :param in_channels: Number of channels for input features
        :param out_channels: Number of channels for output characteristics
        :param num_heads: Number of attention heads
        """
        super().__init__()

        self.num_heads = num_heads
        self.out_channels = out_channels
        self.d = out_channels // num_heads

        # Linear transformation layer for generation of Q, K, V (num_nodes, d) d = out_channels / num_heads (see Transformer)
        self.w_q = nn.Linear(in_channels, out_channels)
        self.w_k = nn.Linear(in_channels, out_channels)
        self.w_v = nn.Linear(in_channels, out_channels)

        # MLP module for final feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(self.d, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, vertex_features, relative_encoding, num_nodes):
        """
        :param num_nodes: Number of nodes
        :param relative_encoding: Relative position encoding, from RSE module
        :param vertex_features: The input node feature is the size of [num_nodes, D_in], where num_nodes is the number of nodes and D_in is the number of input feature channels
        :return:
        """

        # Projection to Q, K, V matrix (num_nodes, d, num_heads)
        q = self.w_q(vertex_features).view(num_nodes, self.d, self.num_heads)
        k = self.w_k(vertex_features).view(num_nodes, self.d, self.num_heads)
        v = self.w_v(vertex_features).view(num_nodes, self.d, self.num_heads)

        k_transposed = k.permute(2, 1, 0)
        q = q.permute(2, 0, 1)

        # Calculate attention_scores
        attention_scores = torch.matmul(q, k_transposed).permute(1, 2, 0)

        # Add relative position encoding
        attention_scores = attention_scores + relative_encoding

        # Softmax calculates weights on each row
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Calculate the weighted value v (num_nodes, num_heads, out_channels)
        v_transposed = v.permute(0, 2, 1)
        attention_output = torch.matmul(attention_scores, v_transposed).sum(dim=1)

        # Merge outputs of all headers(num_nodes, num_heads * out_channels)
        attention_output = attention_output.view(num_nodes, -1)

        output = self.mlp(attention_output)

        return output, attention_scores




