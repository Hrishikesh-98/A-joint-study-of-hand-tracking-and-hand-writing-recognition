import torch
import torch.nn as nn
import torch.nn.functional as F
from Pointnet.models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
import numpy as np

class pointSem(nn.Module):
    def __init__(self):
        super(pointSem, self).__init__()
        self.sa = PointNetSetAbstraction(1024, 0.1, 32, 9, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv = nn.Conv1d(128,16, 1)
        #self.fcx = nn.Linear(1280,512)
        #self.adaptive = nn.AdaptiveAvgPool1d(512)

    def sortit(self,xyz,points):
        print(points.shape, " ", xyz.shape, " ", type(points))
        new_xyz = np.zeros((1,points.shape[2],3))
        new_points = np.zeros((1,points.shape[2],points.shape[1]))
        xyz = xyz.permute(0,2,1).cpu().detach().numpy()
        points = points.permute(0,2,1).cpu().detach().numpy()
        for i,x in enumerate(xyz):
            print(points.shape, " ", x.shape, " ", type(x))
            temp = points[i][np.argsort(x[:,0])].reshape((1,points[1],points[2]))
            new_points = np.append(new_points,temp,axis=0)
            temp = xyz[i][np.argsort(x[:,0])].reshape((1,xyz[1],3))
            new_xyz = np.append(new_xyz,temp,axis=0)
        xyz = torch.tensor(new_xyz).float()
        points = torch.tensor(new_points).float()

        return xyz[1:],permute(0,2,1), points[1:].permute(0,2,1)


    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        print(l0_points.shape, ' ', l0_xyz.shape)

        l1_xyz, l1_points = self.sa(l0_xyz, l0_points)
        #l1_xyz, l1_points = self.sortit(l1_xyz,l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        #l2_xyz, l2_points = self.sortit(l2_xyz,l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        #l3_xyz, l3_points = self.sortit(l3_xyz,l3_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        #l4_xyz, l4_points = self.sortit(l4_xyz,l4_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv(x)
        #x = self.fcx(x)
        #x = self.adaptive(x)
        print("hi")
        print(x.shape)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1).squeeze()
        print(x.shape)
        return x.reshape(-1,16*1280) #l3_points.permute(0,2,1)


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))
