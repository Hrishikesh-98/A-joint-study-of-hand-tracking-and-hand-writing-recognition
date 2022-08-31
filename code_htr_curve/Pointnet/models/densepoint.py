import torch
import torch.nn as nn
from torch.autograd import Variable
from Pointnet.models.pointnet2_utils import PointnetSAModule, PointnetSAModuleMSG
import numpy as np


class DensePoint(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        
        # stage 1 begin
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.25],
                nsamples=[64],
                mlps=[[input_channels, 96]],
                use_xyz=use_xyz,
                pool=True
            )
        )
        # stage 1 end
        
        # stage 2 begin
        input_channels = 96
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.32],
                nsamples=[64],
                mlps=[[input_channels, 93]],
                use_xyz=use_xyz,
                pool=True
            )
        )
        
        input_channels = 93
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.39],
                nsamples=[16],
                mlps=[[input_channels, 96]],
                group_number=2,
                use_xyz=use_xyz,
                after_pool=True
            )
        )
        
        input_channels = 117
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.39],
                nsamples=[16],
                mlps=[[input_channels, 96]],
                group_number=2,
                use_xyz=use_xyz
            )
        )
        
        input_channels = 141
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.39],
                nsamples=[16],
                mlps=[[input_channels, 96]],
                group_number=2,
                use_xyz=use_xyz,
                before_pool=True
            )
        )
        # stage 2 end
       
        # global pooling
        input_channels = 165
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[input_channels, 512], use_xyz=use_xyz
            )
        )


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        print(features.shape)

        return features
