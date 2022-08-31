import torch
import torch.nn as nn
import torch.nn.functional as F
from Pointnet.models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
import numpy as np

class Pointnet2(nn.Module):
    def __init__(self, normal):
        super(Pointnet2, self).__init__()
        in_channel = 3 if normal else 0
        self.normal = normal
        self.sa1 = PointNetSetAbstractionMsg(256, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        #self.sa1 = PointNetSetAbstractionMsg(26, [0.4, 0.8, 1.6], [64, 96, 128], 640,[[128, 128, 256], [256, 256, 512], [256, 256, 512]])
        #self.sa2 = PointNetSetAbstraction(None, None, None, 1280 + 3, [256, 512, 1024, 2048], True)
        self.sa3 = PointNetSetAbstraction(None, None, None, 640+3, [256, 512, 1024], True)
        #self.conv1 = nn.Conv1d(1280, 512, 1)
        #self.bn1 = nn.BatchNorm1d(512)
        #self.conv2 = nn.Conv1d(512, 512, 1)
        #self.bn2 = nn.BatchNorm1d(512)
        #self.conv3 = nn.Conv1d(512, 128, 1)
        #self.bn3 = nn.BatchNorm1d(128)
        #self.drop = nn.Dropout(0.5)
        #self.conv4 = nn.Conv1d(128, 26, 1)
        ##self.bn1 = nn.BatchNorm1d(2048)
        #self.drop1 = nn.Dropout(0.4)
        ##self.fc1 = nn.Linear(2048,2048)
        ##self.bn2 = nn.BatchNorm1d(1024)
        #self.dropp2 = nn.Dropout(0.5)
        ##self.fc2 = nn.Linear(1024, 1024)
        #self.adaptive = nn.AdaptiveAvgPool1d(512)


    def forward(self, xyz):
        B, _, _ = xyz.shape
        ##l2_points = xyz[:, 3:, :]
        ##l2_xyz = xyz[:, :3, :]
        norm = xyz[:,3:,:]
        xyz = xyz[:,:3,:]
        if not self.normal:
            norm = None
        #print(self.normal)
        #print("1")
        #l_xyz, l_points = self.sa(xyz, norm)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        #print("2 ", l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        #print("3 ", l2_points.shape)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        #print("4 ", l3_points.shape)
        #l4_xyz, l4_points = self.sa3(l3_xyz, l3_points)
        #print("5 ", l4_points.shape)
        #l2_points= l2_points.permute(0,2,1)
        #x = self.adaptive(l3_points)
        #l3_points = self.m(l3_points)
        #xyz = np.zeros((1,26,1280))
        #l3_xyz = l3_xyz.permute(0,2,1).cpu().detach().numpy()
        #l3_points = l3_points.permute(0,2,1).cpu().detach().numpy()
        #for i,x in enumerate(l3_xyz):
            #print(x.shape)
            #print(l3_points[i].shape)
            #temp = l3_points[i][np.argsort(x[:,0])].reshape((1,26,1280))
            #print(temp.shape)
            #xyz = np.append(xyz,temp,axis=0)
        x = l3_points.view(B,1024)
        #x2 = l2_points.view(B,2048)
        #x = torch.cat((x1,x2),dim=1)
        ##x = F.relu(self.bn1(self.fc1(x)))
        ##x = self.bn2(self.fc2(x))
        #x = self.fcc1(x)
        #x = x.unsqueeze(2)
        #x = self.fc3(x)
        #x = torch.reshape(l2_points,(B,128*640))
        #print(xyz.shape)
        #x = torch.tensor(xyz).cuda().float()
        #x = self.con1(l3_points)
        #x = self.drop(F.relu(self.bn1(self.fc1(x))))
        #x = self.drop(F.relu(self.bn2(self.conv2(x))))
        #x = self.drop(F.relu(self.bn3(self.conv3(x))))
        #x = self.fc2(x)
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


