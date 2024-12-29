import torch
from torch import optim

from data.cad_dataset import CADDataset
from model.GATCADNet import GATCADNet
from module.NodeEdgeFeatureEnhancer import NodeEdgeFeatureEnhancer
from module.RSE import RSE
import torch.nn as nn

# 数据集
dataset = CADDataset(svg_path='dataset/train')

# 设置训练参数
n_heads = 8
num_epochs = 100
lr = 0.001
beta1 = 0.9
beta2 = 0.99
decay_rate = 0.7
lr_decay_step = 20
lambda_ins = 2

# 损失函数
# 语义损失: CrossEntropyLoss
sem_loss = nn.CrossEntropyLoss()

# 实例损失: BinaryCrossEntropyLoss
ins_loss = nn.BCEWithLogitsLoss(reduction='none')  # 使用none，以便我们可以手动加权

# 模型
model = GATCADNet(
        in_channels=128,
        out_channels=128,
        num_heads=n_heads,
        num_stages=8,
    )

# Adam优化器
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=decay_rate)

# 模型训练过程
for epoch in range(num_epochs):
    model.train()
    for graph in dataset:
        vertex_target = graph.y  # (num_nodes, 1)
        vertex_target_classes = int(vertex_target.max()) + 1
        adj_matrix_target = graph.adj_matrix    # 真实的邻接矩阵
        num_nodes = graph.num_nodes
        vertex_features_list = []

        """ 使用边特征对节点特征进行增强 """
        for node_id in range(graph.x.size(0)):  # 遍历所有的节点
            # (1, 7)
            node_features = graph.x[node_id].unsqueeze(0)  # 获取当前节点的特征

            # 获取与当前节点相连的边的索引
            edge_indices = (graph.edge_index[0] == node_id).nonzero(as_tuple=True)[0]

            # 获取这些边的特征
            # (num_edges（邻边数量）, 7)
            edge_features = graph.edge_attr[edge_indices]
            enhancer = NodeEdgeFeatureEnhancer(node_input_dim=7, edge_input_dim=7, output_dim=128)
            # (1, 128)
            node_new_features = enhancer(node_features, edge_features)
            vertex_features_list.append(node_new_features)

        # 使用torch.stack将列表中的张量拼接成一个(num_nodes, 128)的张量
        vertex_features = torch.stack(vertex_features_list, dim=0).squeeze(1)

        """ 对边特征进行处理，进行相对空间编码 """
        # 创建一个形状为(num_nodes, num_nodes, 7)的张量，用于存储所有边的特征
        edge_feature_matrix = torch.zeros(num_nodes, num_nodes, 7)
        RSE = RSE(in_channels=7, out_channels=n_heads)

        # 将每条边的特征填充到edge_feature_matrix中
        for i in range(graph.edge_index.shape[1]):
            u, v = graph.edge_index[:, i]  # 获取边的节点对(u, v)
            edge_feature_matrix[u, v] = graph.edge_attr[i]  # 填充边特征到(u, v)位置
            edge_feature_matrix[v, u] = graph.edge_attr[i]  # 对于无向图，填充(v, u)位置
        # 使用RSE对该张量进行处理 (num_nodes, num_nodes, 7) -> (num_nodes, num_nodes, n_heads)
        edge_feature_matrix = RSE(edge_feature_matrix)
        num_nodes = edge_feature_matrix.size(0)

        """ 使用数据对GATCADNet进行训练 """
        # 使用模型预测得到的节点特征和邻接矩阵
        # predict_vertex_features: (num_nodes, 30)  predict_adj_matrix: (num_nodes, num_nodes)
        predict_vertex_features, predict_adj_matrix = model(
            vertex_features=vertex_features,
            num_nodes=num_nodes,
            relative_encoding=edge_feature_matrix,
            num_classes=vertex_target_classes
        )

        # 语义损失
        loss_sem = sem_loss(predict_vertex_features, vertex_target)

        # 实例损失
        # 构造权重矩阵
        w = torch.ones(num_nodes, num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if vertex_target[i] == vertex_target[j] and adj_matrix_target[i, j] == 0:
                    w[i, j] = 20
                elif vertex_target[i] == vertex_target[j] and adj_matrix_target[i, j] == 1:
                    w[i, j] = 2
                elif vertex_target[i] != vertex_target[j] and adj_matrix_target[i, j] == 1:
                    w[i, j] = 0

        loss_ins = ins_loss(predict_adj_matrix, adj_matrix_target)

        weighted_loss_ins = loss_ins * w

        final_loss_ins = weighted_loss_ins.mean()

        # 计算总损失
        total_loss = loss_sem + lambda_ins * final_loss_ins

        # 输出损失
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {total_loss.item():.4f}, "
              f"Semantic Loss: {loss_sem.item():.4f}, "
              f"Instance Loss: {final_loss_ins.item():.4f}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # 调整学习率
    if epoch % 20 == 0:
        scheduler.step()
