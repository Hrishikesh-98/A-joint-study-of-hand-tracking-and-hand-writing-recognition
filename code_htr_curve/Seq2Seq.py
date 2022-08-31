from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Pointnet.models.pointnet2_cls_msg_ import Pointnet2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 2

        self.character = ['[SOS]'] + ['[EOS]'] +dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(1)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        #print(length)
        for index, l in enumerate(length):
            #print(text_index, ' ', index, ' ', l)
            t = text_index[index]

            char_list = []
            for i in range(l):
                if t[i] == 1:
                    break
                if t[i] != 0 and t[i] != 1:
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers,bidirectional):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n = num_layers
        self.bi = 1
        if bidirectional:
            self.bi = 2
        self.gru = nn.GRU(input_size, hidden_size,batch_first=True, num_layers= num_layers, bidirectional=bidirectional)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.n*self.bi, batch_size, self.hidden_size, device=device)

class EncoderFFN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderFFN, self).__init__()
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(3072,2048)
        self.fc2 = nn.Linear(2048,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512,256)
        self.fc5 = nn.Linear(256,128)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.5)

    def forward(self, input):
        x = self.drop(F.relu(self.bn1(self.fc1(input))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        x = self.drop(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x.view(x.size()[0],1,128)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(1, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        #print(hidden.size())
        output, hidden = self.gru(input, hidden[:,:1,:])
        output = self.out(output)
        #print(output.size())
        output = self.softmax(output)
        #print(output.size())
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=25):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.attn = nn.Linear(self.hidden_size +1, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size +1, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        attn_weights = F.softmax(
            self.attn(torch.cat((input[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=1280):
    encoder_hidden = encoder.initHidden()
    #encoder_hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)
    #print(encoder_hidden.size())


    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    batch_size = input_length
    target_length = 25
    bi = 1
    #print(target_length)
    if bidirectional:
        bi = 2
    encoder_hidden = torch.zeros(num_layers*bi, batch_size, 128, device=device)

    #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    
    decoded_text = [ [1]*25]*batch_size

    '''for ei in range(input_length):
        print(input_tensor[ei].size())
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]'''

    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

    #encoder_hidden = encoder(input_tensor.float())

    decoder_input = torch.tensor(batch_size*[[0]], device=device).float()
    decoder_input = torch.zeros(batch_size,1,1, device=device).float()

    decoder_hidden = encoder_hidden[-1].unsqueeze(0)
    length = 0
    #print(decoder_hidden.size())

    use_teacher_forcing =  True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            #print(target_tensor[:,di].size(), ' ', decoder_input.size())
            decoder_output, decoder_hidden = decoder(
                decoder_input.float(), decoder_hidden) #, encoder_outputs)
            #loss += criterion(decoder_output.squeeze(0), target_tensor[di].unsqueeze(0))
            decoder_input = target_tensor[:,di].view(batch_size,1,1)  # Teacher forcing
            topv, topi = decoder_output.topk(1)
            loss += criterion(decoder_output.squeeze(1), target_tensor[:,di])
            for i in range(batch_size):
                decoded_text[i][di] = topi[i].item()


    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            #print("once", ' ', decoder_input.size())
            decoder_output, decoder_hidden = decoder(
                decoder_input.float(), decoder_hidden) #, encoder_outputs)
            #print(decoder_output.size(), ' ', target_tensor[:,di].size(), ' ', target_tensor.size(), ' ', target_tensor[:,di])
            topv, topi = decoder_output.topk(1)
            #print('topi ',topi.size(), ' ', topi )
            decoder_input = topi.detach()  # detach from history as input
            #print(decoder_input.item)
            loss += criterion(decoder_output.squeeze(1), target_tensor[:,di])
            for i in range(batch_size):
                decoded_text[i][di] = topi[i].item()
    #print(decoded_text)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return decoded_text, loss.item()
    
    
def evaluate(encoder, decoder,input_tensor, max_length=25):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        batch_size = input_length
        encoder_hidden = encoder.initHidden()
        bi = 1
        #print(target_length)
        if bidirectional:
            bi = 2
        encoder_hidden = torch.zeros(num_layers*bi, batch_size, 128, device=device)

        #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        '''for ei in range(input_length):
            #print(input_tensor[ei].size())
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]'''

        #encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        encoder_output, encoder_hidden = encoder(input_tensor.float(),encoder_hidden)

        decoder_input = torch.tensor([[0]], device=device).float()
        decoder_input = torch.zeros(batch_size,1,1, device=device).float()

        decoder_hidden = encoder_hidden[-1].unsqueeze(0)

        decoded_words =  [ [1]*25]*batch_size

        '''for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('[EOS]')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()'''

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input.float(), decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input
            for i in range(batch_size):
                #print(topi[i].item())
                decoded_words[i][di] = topi[i].item()



        return decoded_words


def trainIters(encoder, decoder, data, labels, data_val, labels_val, converter, n_iters, print_every=1000, plot_every=100, learning_rate=0.1):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    prev_count  = 0
    prev_train = 0

    for iter in range(1, n_iters + 1):
        print(iter)
        loss = 0
        count  = 0
        for i, batch in enumerate(data):
            input_tensor = batch
            batch_size = input_tensor.size()[0]
            #print(input_tensor.size())
            label = labels[i*batch_size:i*batch_size+batch_size]
            input_tensor = input_tensor.to(device)
            target_tensor, length = converter.encode(label, batch_max_length=25)
            #print(target_tensor.size())

            ret_text,l = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            pred = converter.decode(ret_text,[25]*batch_size)
            for k in range(batch_size):
                if pred[k] == label[k]:
                    count = count +1

            if prev_train <= count:
                prev_train  = count

            print('epoch ', iter , ' ' ,i , ' ', labels[i],' ', pred[0], ' ', prev_count, ' ', prev_train, ' ', l)
            loss += l

        print_loss_total += loss/49494

        count  = 0

        for i, batch in enumerate(data_val):
            input_tensor = batch
            batch_size = input_tensor.size()[0]
            label = labels_val[i*batch_size:i*batch_size+batch_size]
            input_tensor = input_tensor.to(device)
            target_tensor, length = converter.encode(label, batch_max_length=25)
            ret_text = evaluate(encoder,decoder,input_tensor)
            pred = converter.decode(ret_text,[25]*batch_size)
            print(i, ' ', labels_val[i],' ', pred[0])
            for k in range(batch_size):
                if pred[k] == label[k]:
                    count = count +1

            if prev_count <= count:
                prev_count  = count
                torch.save(
                encoder.state_dict(), f'./saved_models/seq2seq/best_encoder_batch.pth')
                torch.save(
                decoder.state_dict(), f'./saved_models/seq2seq/best_decoder_batch.pth')



hidden_size = 128
input_size = 7
batch_size = 32
num_layers = 5
bidirectional = True

#feats = np.load('/net/voxel03/misc/extra/data/hrishikesh/feats_train.npy')
#sem_rep = np.load('/net/voxel03/misc/extra/data/hrishikesh/semantic_rep_train.npy')
data = np.load('/net/voxel03/misc/extra/data/hrishikesh/points_normals_train.npy')
data = np.load('/net/voxel03/misc/extra/data/hrishikesh/curves_train.npy')
data = data.reshape((data.shape[0],128,7))
print(data.shape)
#data = np.hstack((feats,sem_rep))
#labels = np.load('/net/voxel03/misc/extra/data/hrishikesh/text_normals_trainx2.npy')
labels = np.load('/net/voxel03/misc/extra/data/hrishikesh/curve_text_train.npy')
#labels = np.loadtxt('/net/voxel03/misc/extra/data/hrishikesh/curve_text_train.txt',dtype=str)
data = torch.tensor(data)
data = DataLoader(data, batch_size=batch_size)


#feats = np.load('/net/voxel03/misc/extra/data/hrishikesh/feats_val.npy')
#sem_rep = np.load('/net/voxel03/misc/extra/data/hrishikesh/semantic_rep_val.npy')
#data_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/points_normals_val.npy')
#data_val = data_val.reshape((data_val.shape[0],1280,6))
data_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/curves_val.npy')
data_val = data_val.reshape((data_val.shape[0],128,7))
print(data_val.shape)
#data_val = np.hstack((feats,sem_rep))
#labels_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/text_normals_val.npy')[:7380]
labels_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/curve_text_val.npy')
data_val = torch.tensor(data_val)
data_val = DataLoader(data_val, batch_size=batch_size)

characters = string.printable[:-6]
converter = LabelConverter(characters)
output_size = len(converter.character)

encoder1 = EncoderRNN(input_size, hidden_size, num_layers, bidirectional).to(device)
#encoder1 = Pointnet2().to(device)
decoder1 = DecoderRNN(hidden_size, output_size).to(device)
#decoder1 = AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)

#print(labels.shape, ' ', labels[0])
trainIters(encoder1, decoder1, data, labels,data_val,labels_val, converter, 7500, print_every=5000,learning_rate=0.001)
