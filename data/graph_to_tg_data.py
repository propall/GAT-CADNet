import torch
import networkx as nx
from torch_geometric.data import Data
import numpy as np


# 将networkx的图转换为torch_geometric的Data类
def convert_nx_to_tg_data(graph: nx.Graph) -> Data:
    # 获取graph中所有点的特征，拼接成顶点特征向量
    node_features = [data['features'] for _, data in graph.nodes(data=True)]
    node_features = np.array(node_features).astype(np.float32)

    edge_features = [data['features'] for _, _, data in graph.edges(data=True)]
    edge_features = np.array(edge_features).astype(np.float32)

    # 获取边的连接信息(edge_index)
    edge_index = np.array(list(graph.edges)).T  # shape: (2, num_edges)

    # 获取target，即正确的节点分类
    target = [data.get('target') for _, data in graph.nodes(data=True)]

    # 将数据转换为pytorch tensor
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    edge_features_tensor = torch.tensor(edge_features, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.int64)

    # 创建torch_geometric的Data对象
    data = Data(
        x=node_features_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_features_tensor,
        y=target_tensor
    )

    # 计算邻接矩阵
    adj_matrix = torch.zeros(data.num_nodes, data.num_nodes)
    for i in range(edge_index_tensor.size(1)):
        src, dst = edge_index_tensor[:, i]
        adj_matrix[src, dst] = 1
        adj_matrix[dst, src] = 1

    data.adj_matrix = adj_matrix

    return data





