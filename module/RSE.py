"""
RSE module: (Relative spatial encoding): 相对空间编码
研究人员经常使用相对位置编码来使网络对平移不变并感知距离
使用多层感知机MLP来对初始边特征进行处理以编码顶点之间的相对空间关系
"""
from torch import nn
import torch.nn.functional as F


class RSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


