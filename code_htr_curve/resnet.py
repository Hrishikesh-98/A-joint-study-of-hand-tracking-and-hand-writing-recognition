import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        

class layer(nn.Module):
    def __init__(self,block, in_channel, out_channel, pooling, initial=0):
        super(layer, self).__init__()
        self.conv1 = block(in_channel, in_channel)
        self.conv2 = block(in_channel, in_channel)
        self.conv3 = block(in_channel, out_channel)
        #self.avgPool = nn.AdaptiveAvgPool1d(pooling)
        if initial == 0:
            self.fc = nn.Linear(pooling*2,pooling)
        else:
            self.fc = nn.Linear(initial,pooling)

    def forward(self, x):
        #pdb.set_trace()
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out)

        return out

        
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = layer(BasicBlock,1, 2,2048)
        self.conv2 = layer(BasicBlock,2, 4,1024)
        self.conv3 = layer(BasicBlock,4, 8,512)
        self.conv4 = layer(BasicBlock,8, 16,256)
        self.conv5 = layer(BasicBlock,16, 26,128)
        #self.pred = nn.Linear(128,95)

    def forward(self, x, roi=None):
        #pdb.set_trace()
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        #out = self.pred(out)
        
        return out

class ResNet2(nn.Module):
    def __init__(self):
        super(ResNet2, self).__init__()
        self.conv1 = layer(BasicBlock,1, 2,2048, 3072)
        self.conv2 = layer(BasicBlock,2, 4,1024)
        self.conv3 = layer(BasicBlock,4, 8,512)
        self.conv4 = layer(BasicBlock,8, 16,256)
        self.conv5 = layer(BasicBlock,16, 26,128)
        #self.pred = nn.Linear(128,95)

    def forward(self, x, roi=None):
        #pdb.set_trace()
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        #out = self.pred(out)

        return out

