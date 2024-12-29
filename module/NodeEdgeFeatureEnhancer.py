import torch
import torch.nn as nn
import torch.nn.functional as F


# 用于对节点特征和边特征进行特征提取
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class NodeEdgeFeatureEnhancer(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim):
        super().__init__()

        # MLP提取节点特征和边特征
        self.node_mlp = MLP(node_input_dim, 64)     # 将节点特征映射到64维
        self.edge_mlp = MLP(edge_input_dim, 64)     # 将边特征映射到64维

        # 最终输出维度
        self.output_dim = output_dim

    def forward(self, node_features, edge_features):
        # 嵌入节点特征
        node_emb = self.node_mlp(node_features)     # 节点嵌入 v_hat_i

        # 嵌入边特征
        edge_emb = self.edge_mlp(edge_features)     # 边嵌入 e_hat_ij

        # 对每个节点的边特征进行池化(max pooling)
        if edge_emb.size(0) > 0:
            pooled_edge_feats = torch.max(edge_emb, dim=0)[0].unsqueeze(0)
        else:
            pooled_edge_feats = torch.zeros(edge_emb.size(1)).unsqueeze(0)  # 处理为空的情况

        # 拼接节点特征和池化后的边特征
        enhanced_node_features = torch.cat([node_emb, pooled_edge_feats], dim=1)

        return enhanced_node_features

