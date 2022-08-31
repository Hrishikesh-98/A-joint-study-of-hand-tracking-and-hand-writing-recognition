import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from Pointnet.models.pointnet2_cls_ssg_ import Pointnet_ssg
from Pointconv.model.pointconv import PointConvDensityClsSsg as Pointconv
from Pointnet.models.pointnet2_cls_msg_ import Pointnet2 as Pointnet
from Pointnet.models.pointnet2_cls_msg__ import Pointnet2 as PointnetNew
from Pointnet.models.pointnet2_cls_msg___ import Pointnet2 as PointnetOnly
from Pointnet.models.pointnet2_sem_seg_ import pointSem
from Pointnet.models.pointnet2_sem_seg_msg_ import get_model
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from hwnet.pytorch import resnetROI, resnet
from resnet import ResNet, ResNet2
from resnet_feature import ResNet_FeatureExtractor as Res
from transformer3 import Transformer
import torchvision.models as models
from custom_conv import Custom_conv
from resnet_baseline import ResNetBaseline

#from Pointnet.models.densepoint import DensePoint
from Prediction import Attention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True,dropout=0.5)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class UnidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(UnidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class ResCNN(nn.Module):

    def __init__(self, in_channel, out_channel, kernel):
        super(ResCNN, self).__init__()
        self.cnn1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel, stride=1,padding=1)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.cnn2 = nn.Conv1d(in_channels=out_channel, out_channels=2*out_channel, kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(2*out_channel)
        self.maxPool = nn.MaxPool2d(2, stride=2)
        #self.linear = nn.Linear(linear, int(linear/2))
        #self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        out = F.relu(self.bn1(self.cnn1(input)))  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        out = F.relu(self.bn2(self.cnn2(out)))
        out = self.maxPool(out)
        #out = F.relu(self.linear(out))
        return out


class CNN(nn.Module):

    def __init__(self, in_channel, out_channel, kernel):
        super(CNN, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel, stride=1)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        convolutional = F.relu(self.bn(self.cnn(input) ) )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        return convolutional



class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt

        self.FeatureExtraction = models.resnet101(pretrained=True)
        self.FeatureExtraction = nn.Sequential(*list(self.FeatureExtraction.children())[:-4])

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(opt.input_size, opt.hidden_size, opt.hidden_size),
            #BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            #BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            #BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.SequenceModeling_output = opt.hidden_size

        """ Prediction """
        #if opt.Prediction == 'CTC':
        self.Prediction = nn.Linear(self.SequenceModeling_output, opt.output_size)
        #elif opt.Prediction == 'Attn':
        #    self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)

    def forward(self, input, is_train=True):

        """ Sequence modeling stage """

        features = self.FeatureExtraction(input).permute(0,2,3,1)
        #print(features,shape)
        features = torch.reshape(features,(features.shape[0], features.shape[1]*features.shape[2],features.shape[3]))
        #print(features.shape)

        contextual_feature = self.SequenceModeling(features)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction



class LSTM(nn.Module):

    def __init__(self, opt):
        super(LSTM, self).__init__()
        self.opt = opt

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(opt.input_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))

        self.SequenceModeling_output = opt.hidden_size

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)


    def forward(self, input, text,is_train=True):


        #hw = self.fc1(hw)

        features = input


        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(features)

        """ Prediction stage """
        if self.opt.Prediction == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction



class PointHWnet(nn.Module):

    def __init__(self, opt):
        super(PointHWnet, self).__init__()
        self.opt = opt

        """feature extraction"""
        self.FeatureExtraction_point = nn.Sequential(
            CNN(1, 1, 1536),
            CNN(1, 1, 768),
            CNN(1, 1, 384),
            CNN(1, 1, 192),
            CNN(1, 1, 96),
            CNN(1, 1, 48))

        #self.FeatureExtraction_hw = nn.Sequential(
        #    CNN(1,1,1024),
        #    CNN(1, 1, 512),
        #    CNN(1, 1, 256),
        #    CNN(1, 1, 192))

        #self.m = nn.MaxPool2d(2, stride=2)

        #self.fc1 = nn.Linear(2048, 1024)

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(opt.input_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.SequenceModeling_output = opt.hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)

    def forward(self, input, feats,is_train=True):

        features = torch.cat((input,feats),1)
        #feats = feats.unsqueeze(1)
        #input = input.unsqueeze(1)

        """Feature Extraction"""
        #features_point = self.FeatureExtraction_point(input)
        #features_hw = self.FeatureExtraction_hw(feats)
        #features_point = self.m(features_point)
        #features_hw = self.m(features_hw)

        """Augmentation"""
        #features = features_point + features_hw
        features = features.unsqueeze(1)
        features = self.FeatureExtraction_point(features)
        features = features.squeeze()
        #features = torch.reshape(features,(features.shape[0],32,32))
        #features = torch.cat(feats,input)
        features = features.unsqueeze(2)
        #print(features.shape)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(features)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction

class NewModel(nn.Module):

    def __init__(self, opt):
        super(NewModel, self).__init__()
        self.opt = opt

        self.FeatureExtraction = Res(1,512)

        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(opt.input_size, opt.hidden_size, opt.hidden_size),
            #BidirectionalLSTM(opt.input_size, opt.hidden_size, opt.hidden_size),
            #BidirectionalLSTM(opt.input_size, opt.hidden_size, opt.hidden_size),
            #BidirectionalLSTM(opt.input_size, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.SequenceModeling_output = opt.hidden_size

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)

    def forward(self, input, is_train=True):

        """ Sequence modeling stage """

        features = self.FeatureExtraction(input)
        print(features.shape)
        features = self.AdaptiveAvgPool(features.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        features = features.squeeze(3)

        contextual_feature = self.SequenceModeling(features)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction


class Mdl6(nn.Module):

    def __init__(self, opt):
        super(Mdl6, self).__init__()
        self.opt = opt

        """Feature Extraction"""
        self.PointnetFeatureExtraction = Pointnet()
        self.HwnetFeatureExtraction = resnetROI.ResNetROI34() #PointnetNew() #get_model() #pointSem()

        self.fc1 = nn.Linear(4096,2048)
        self.fc2 = nn.Linear(2048,2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)

        """ Prediction """
        self.Pred = nn.Linear(2048, 7585)

    def forward(self, input, feats,is_train=True):

        '''FeatureExtraction'''
        inputs, targets, roi = feats['image'].to(device), feats['label'].to(device), feats['roi'].to(device)
        roi[:,0] = torch.arange(0,roi.size()[0])
        inputs = inputs.unsqueeze(1)
        features_points = self.PointnetFeatureExtraction(input)
        output, features_hwnet = self.HwnetFeatureExtraction(inputs,roi)
        features = torch.cat((features_hwnet,features_points),dim=1)
        features = F.relu(self.bn1(self.fc1(features)))
        Outfeatures = self.bn2(self.fc2(features))
        features = F.relu(Outfeatures)

        """ Prediction stage """
        prediction = self.Pred(features)
        return prediction , features


class Mdl7(nn.Module):

    def __init__(self, opt):
        super(Mdl7, self).__init__()
        self.opt = opt

        """Feature Extraction"""
        self.PointnetFeatureExtraction =  PointnetOnly()
        #self.HwnetFeatureExtraction = resnetROI.ResNetROI34() #PointnetNew() #get_model() #pointSem()
        #self.hwfc = nn.Linear(3072,2048)
        #self.hwbn = nn.BatchNorm1d(2048)
        self.fc1 = nn.Linear(3072,2048)
        self.fc2 = nn.Linear(2048,2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)

        """ Prediction """
        self.Pred = nn.Linear(2048, 7002)

    def forward(self, input, feats,is_train=True):

        '''FeatureExtraction'''
        #print(input.shape)
        features_points = self.PointnetFeatureExtraction(input)
        features_hwnet = feats #F.relu(self.hwbn(self.hwfc(feats))) #torch.tensor(np.zeros((input.shape[0],2048))).float().to(device)
        features = torch.cat((features_hwnet,features_points),dim=1)
        features = F.relu(self.bn1(self.fc1(features)))
        Outfeatures = self.bn2(self.fc2(features))
        features = F.relu(Outfeatures)

        """ Prediction stage """
        prediction = self.Pred(features)

        return prediction , Outfeatures

class Mdl9(nn.Module):

    def __init__(self, opt):
        super(Mdl9, self).__init__()
        self.opt = opt

        """Feature Extraction"""
        self.PointnetFeatureExtraction = models.resnet34()
        self.PointnetFeatureExtraction.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True,dropout=0.5)
        #self.HwnetFeatureExtraction = resnetROI.ResNetROI34() #PointnetNew() #get_model() #pointSem()

        self.fc1 = nn.Linear(1000,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(1000)

        """ Prediction """
        self.Pred = nn.Linear(1000, 7002)

    def forward(self, input, feats,is_train=True):

        '''FeatureExtraction'''
        print(input.shape)
        features = self.PointnetFeatureExtraction(input.unsqueeze(1))
        features = F.relu(self.bn1(self.fc1(features)))
        Outfeatures = self.bn2(self.fc2(features))
        features = F.relu(Outfeatures)

        """ Prediction stage """
        prediction = self.Pred(features)

        return prediction , Outfeatures

class Mdl16(nn.Module):

    def __init__(self, opt):
        super(Mdl16, self).__init__()
        self.opt = opt

        """Feature Extraction"""
        self.PointnetFeatureExtraction = PointnetNew() #ResNetBaseline(3) #Custom_conv(3) #Pointconv()
        #self.HwnetFeatureExtraction = resnetROI.ResNetROI34() #PointnetNew() #get_model() #pointSem()

        self.fc1 = nn.Linear(3072,2048)
        self.fc2 = nn.Linear(2048,2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(2048)

        """ Prediction """
        self.Pred = nn.Linear(2048, 7003)

    def forward(self, input, points, feats,is_train=True):

        '''FeatureExtraction'''
        #print(input.shape)
        features_points = self.PointnetFeatureExtraction(input)
        features_hwnet = feats
        features = torch.cat((features_hwnet,features_points),dim=1)
        #Outfeatures = features
        features = F.relu(self.bn1(self.fc1(features)))
        Outfeatures = self.bn2(self.fc2(features))
        features = F.relu(Outfeatures)

        """ Prediction stage """
        prediction = self.Pred(features)

        return prediction , Outfeatures
