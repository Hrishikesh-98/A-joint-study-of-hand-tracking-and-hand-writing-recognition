import torch.nn as nn
import torch.nn.functional as F
from Pointnet.models.pointnet2_utils import PointNetSetAbstraction
import numpy as np

class Pointnet_ssg(nn.Module):
    def __init__(self,normal_channel=True):
        super(Pointnet_ssg, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=128, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=128, radius=0.6, nsample=64, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=32, radius=0.8, nsample=128, in_channel=512 + 3, mlp=[512, 512, 1024], group_all=False)
        self.sa5 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=1024 + 3, mlp=[1024,1024, 2048], group_all=True)
        self.fc1 = nn.Linear(2048, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(2048, 2048)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            xyz = xyz[:,:3,:]
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        x = l5_points.view(B, 2048)
        #print("hahahahahahahahaha")
        #x = l3_points.permute(0,2,1)
        #l3_points = l3_points.permute(0,2,1).cpu().detach().numpy()
        #for i,x in enumerate(l3_xyz):
            #print(x.shape)
            #print(l3_points[i].shape)
            #temp = l3_points[i][np.argsort(x[:,0])].reshape((1,32,1280))
            #print(temp.shape)
            #xyz = np.append(xyz,temp,axis=0)
        #x = l3_points.view(B,1024)
        #x = F.relu(self.bnn1(self.fcc1(x)))
        #x = F.relu(self.bnn2(self.fcc2(x)))
        #x = self.fcc1(x)
        #x = x.unsqueeze(2)
        #x = self.fc3(x)
        #x = torch.reshape(l3_points,(B,32,64))
        #print(xyz.shape)
        #x = torch.tensor(xyz).cuda().float()
        #x = self.fcc1(x[1:])

        #print(l3_points.shape)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        #x = F.log_softmax(x, -1)


        return x



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
