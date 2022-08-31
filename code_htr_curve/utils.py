import torch
import numpy as np
import csv
import random
from collections import OrderedDict
import json
from nltk.stem.porter import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_json(model, path):
  data_dict = OrderedDict()
  with open(path, 'r') as f:
    data_dict = json.load(f)
  own_state = model.state_dict()
  for k, v in data_dict.items():
    print('Loading parameter:', k)
    if not k in own_state:
      print('Parameter', k, 'not found in own_state!!!')
      continue
    if type(v) == list or type(v) == int:
      v = torch.tensor(v)
    own_state[k].copy_(v)
  model.load_state_dict(own_state)
  print('Model loaded')



def load_npy(type,num):
    y = np.zeros((1,128*640))
    for i in range(1,num):

        x = np.load('./dataset/semantic_rep_'+type+'_'+str(i)+'.npy')
        print(x.shape)
        y = np.append(y,x,axis=0)
        print(y.shape)
        x = input("do you want to continue")

    return y[1:]



def load_csv(file1,file2,twod=True, i=0):
    with open(file1, 'r') as f:
        reader = csv.reader(f)
        x = list(reader)

    with open(file2, 'r') as f:
        reader = csv.reader(f)
        y = list(reader)


    new_y = []
    for row in y:
        str1 = ''
        new_y.append(str1.join(row))

    new_x = []
    print(len(x))
    for row in x:
        if twod:
            nwrow = []
            for r in row:
                nwrow.append(np.array(r[1:-4].split()))
            new_x.append(nwrow)
        else:
            new_x.append(np.float32(row))
        #print(type(np.float32(row)[0]))

    if i == 0:
        joined_lists = list(zip(np.float64(new_x), new_y))
        random.shuffle(joined_lists) # Shuffle "joined_lists" in place
        new_x, new_y = zip(*joined_lists) # Undo joining

    return np.float64(new_x), new_y

class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res



class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

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
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        #print(len(length), ' ', len(text_index))
        for index, l in enumerate(length):
            #if index == len(text_index):
            #    break
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            if index == text_index.shape[0]:
                break
            text = ''.join([self.character[i] for i in text_index[index, :]])
            text_EOS = text.find('[s]')
            texts.append(text[1:text_EOS])
        return texts




def getVocab(ann_file, portStem=False):
    
    vocab = {}
    vocabIdx = {}

    vCntr=0
    with open(ann_file) as vFile:
        for line in vFile:
            tempStr = line.split()
            v = tempStr[1].lower()
            if v not in vocabIdx:
                vocabIdx[vCntr] = v

                vCntr = vCntr +1
    print(vCntr)
    return vocabIdx


def sortTheArray(points, type):
    ret = np.zeros((1,1024,3))
    for i in range(points.shape[0]):
        temp = points[i][np.argsort(points[i][:,0])].reshape((1,1024,3))
        ret = np.append(ret,temp,axis=0)
        print(i)
    np.save("../../../extra/data/hrishikesh/sample_points_image_1024_sorted_"+type+".npy",ret[1:])
    return ret[1:]

def ColorTheArray(points, type):
    colorMap = {0 : [0,0,0], 1 : [255,255,255], 2: [255,0,0] , 3 : [0,255,0] , 4 : [0,0,255] , 5 : [128, 64, 0] , 6 : [ 0,82, 64], 7 : [17,0,128] ,
                8 : [91,19,61], 9 : [192,76,47], 10: [128,192,64] , 11 : [128,0,192] , 12 : [64,192,255] , 13 : [64, 128, 10] , 14 : [ 128,128, 128], 15 : [64,64,64] ,
                16 : [192,192,192], 17 : [100,0,100], 18: [100,100,0] , 19 : [0,100,100] , 20 : [32,192,64] , 21 : [64, 255, 192] , 22 : [ 255,10, 90], 23 : [48,31,45] }
    ret = np.zeros((points.shape[0],1024,6))
    for i in range(points.shape[0]):
        ret[0] = points[0]
        ret[1] = points[1]
        ret[2] = points[2]
        ret[3] = colorMap[points[3]][0]
        ret[4] = colorMap[points[3]][1]
        ret[5] = colorMap[points[3]][2]

        #print(i)
    #np.save("../../../extra/data/hrishikesh/sample_points_image_1024_colored_"+type+".npy",ret[1:])
    return ret
