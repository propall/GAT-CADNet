"""
RSE module: (Relative spatial encoding): Relative space encoding
Researchers often use relative position coding to keep the network translation unchanged and perceive distance
Use multi-layer perceptron MLP to process initial edge features to encode relative spatial relationships between vertices
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


