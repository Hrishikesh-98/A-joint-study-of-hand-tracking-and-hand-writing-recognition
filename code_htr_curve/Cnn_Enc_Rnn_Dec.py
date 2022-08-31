from torch import Tensor
import torch.nn.functional as F
import torch
from torch import nn
import torchvision.models as models
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderCNN(nn.Module):
    def __init__(self, embed_size = 1024):
        super(EncoderCNN, self).__init__()
        
        # get the pretrained densenet model
        #self.densenet = models.densenet121(pretrained=True)
        self.resnet = models.resnet34(pretrained=True)
        #self.resnet = nn.Sequential(*list(models.resnet101().children())[:-4])        
        # replace the classifier with a fully connected embedding layer
        self.resnet.fc = nn.Linear(512, 256)
        
        # add another fully connected layer
        #self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        #self.fc2 = nn.Linear(in_features=1024, out_features = 512)
        #self.fc3 = nn.Linear(in_features=512, out_features = 256)
        #self.fc4 = nn.Linear(in_features=256, out_features = 128)
        #self.bn1 = nn.BatchNorm1d(1024)
        #self.bn2 = nn.BatchNorm1d(512)
        #self.bn3 = nn.BatchNorm1d(256)
        #self.bn4 = nn.BatchNorm1d(128)
        #self.fc_out = nn.Linear(in_features=1024, out_features=embed_size)

        # dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
        # activation layers
        self.relu = nn.ReLU()
        
    def forward(self, word_image):
        
        # get the embeddings from the densenet
        #densenet_output = self.dropout(F.relu(self.densenet(word_image)))
        features = self.dropout(F.relu(self.resnet(word_image)))
        #features = torch.cat((densenet_output,resnet_output),dim=1)
        # pass through the fully connected
        #features = F.relu(self.bn1(self.fc1(word_image)))
        #features = F.relu(self.bn2(self.fc2(features)))
        #features = F.relu(self.bn3(self.fc3(features)))
        #features = F.relu(self.bn4(self.fc4(features)))
        #features = self.fc_out(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,output_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # lstm cell
        #self.lstm_first_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        self.dec = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self, features, point_cloud, max_seq_len, is_train = True):
        
        # batch size
        batch_size = features.size(0)
        
        # init the hidden and cell states to zeros
        #hidden_state = torch.zeros((batch_size,self.hidden_size)).to(device)
        hidden_state = features
        initial = torch.zeros((batch_size,2)).to(device)
        cell_state = torch.zeros((batch_size, self.hidden_size)).to(device)
        # define the output tensor placeholder
        outputs = torch.zeros((batch_size, 1, self.output_size)).to(device)
        #print(outputs.shape)

        # pass the caption word by word
        if is_train:
            for t in range(max_seq_len):
                #print(t, ' ' , initial.shape, ' ', cell_state.shape, ' ', features.shape
                if t == 0:
                    hidden_state, cell_state = self.dec(point_cloud[:,t,:], (features, cell_state))
                # for the 2nd+ time step, using teacher forcer
                else:
                    hidden_state, cell_state = self.dec(point_cloud[:,t,:], (hidden_state,cell_state))
                # output of the attention mechanism
                out = self.fc(hidden_state)
                # build the output tensor
                outputs = torch.cat((outputs,out.unsqueeze(0)), dim=1)
                #outputs[:, t, :] = out
                
        else:
            t = 0
            #if t == 0:
            #    cond = torch.ones((1,1,6))
            #else:
                #cond = outputs[:,t-1,:]
            out = outputs[0]
            while (not (torch.equal(out, torch.tensor([[256, 256]]).float().cuda()))) and t < max_seq_len:
                #print(t)

                # for the first time step the input is the feature vector
                if t == 0:
                    hidden_state, cell_state = self.dec(initial, (features, cell_state))

                # for the 2nd+ time step, using teacher forcer
                else:
                    hidden_state, cell_state = self.dec(out, (hidden_state,cell_state))

                # output of the attention mechanism
                out = self.fc(hidden_state)

                #print(out.shape)
                # build the output tensor
                outputs = torch.cat((outputs,out.unsqueeze(0)), dim=1)
                #outputs[:, t, :] = out

                t += 1
            if t < max_seq_len:
                points_size = outputs.shape[1]
                pad = torch.tensor(np.array([[[256,256]]*(max_seq_len-points_size)])).cuda()
                print(pad.shape, ' ', outputs.shape)
                outputs = torch.cat((outputs,pad),dim=1)
        #print(outputs.shape)
        return outputs[:,1:,:]
        
class EncDec(nn.Module):
    def __init__(
        self, 
        encoder_embed_size = 256,
        input_size = 2,
        decoder_embed_size = 256,
        hidden_size = 256,
        output_size = 2,
        num_layers = 1,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.encoder = EncoderCNN(
            embed_size = encoder_embed_size
        )
        self.decoder = DecoderRNN(
            input_size=input_size,
            embed_size=decoder_embed_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
        )

    def forward(self, word_image, point_cloud, max_seq_len, is_train) -> Tensor:
        #print(hand_track.shape)
        return self.decoder(self.encoder(word_image), point_cloud, max_seq_len, is_train)

