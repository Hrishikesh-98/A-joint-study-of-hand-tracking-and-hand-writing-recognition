import torch
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class Custom_conv(nn.Module):
    def __init__(self, input_channel, output_channel=128):
        super(Custom_conv, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv1d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool1d(4, 4),  # 64x16x50
            nn.Conv1d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool1d(2, 2),  # 128x8x25
            nn.Conv1d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # 256x8x25
            nn.Conv1d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool1d(2, 2),  # 256x4x25
            nn.Conv1d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm1d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
            nn.Conv1d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm1d(self.output_channel[3]), nn.ReLU(True),
            nn.BatchNorm1d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool1d(2, 2),  # 512x2x25
            nn.Conv1d(self.output_channel[3], self.output_channel[3], 1, 1, 0), nn.ReLU(True))  # 512x1x24
            
        self.fc1 = nn.Linear(4096,2048)
        self.fc2 = nn.Linear(2048,2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)
        
    def forward(self, x):
        for layer in self.ConvNet:
            x = layer(x)
        x = x.reshape((-1,128*32))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return x