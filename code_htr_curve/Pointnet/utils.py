import torch
import numpy as np
import csv
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_csv(file1,file2):
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
    for row in x:
        nwrow = []
        for r in row:
            nwrow.append(np.array(r[1:-1].split()))
        new_x.append(nwrow)

    #joined_lists = list(zip(np.float64(new_x), new_y))
    #random.shuffle(joined_lists) # Shuffle "joined_lists" in place
    #new_x, new_y = zip(*joined_lists) # Undo joining

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
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts

def ColorTheArray(points, type):
    colorMap = {0 : [0,0,0], 1 : [255,255,255], 2: [255,0,0] , 3 : [0,255,0] , 4 : [0,0,255] , 5 : [128, 64, 0] , 6 : [ 0,82, 64], 7 : [17,0,128] ,
                8 : [91,19,61], 9 : [192,76,47], 10: [128,192,64] , 11 : [128,0,192] , 12 : [64,192,255] , 13 : [64, 128, 10] , 14 : [ 128,128, 128], 15 : [64,64,64] ,
                16 : [192,192,192], 17 : [100,0,100], 18: [100,100,0] , 19 : [0,100,100] , 20 : [32,192,64] , 21 : [64, 255, 192] , 22 : [ 255,10, 90], 23 : [48,31,45], 24 : [9, 27, 81] }
    ret = np.zeros((points.shape[0],1024,6))
    for i in range(points.shape[0]):
        for j in range(1024):
            ret[i][j][0] = points[i][j][0]
            ret[i][j][1] = points[i][j][1]
            ret[i][j][2] = points[i][j][2]
            ret[i][j][3] = colorMap[points[i][j][3]][0]
            ret[i][j][4] = colorMap[points[i][j][3]][1]
            ret[i][j][5] = colorMap[points[i][j][3]][2]

        print(i)

    print(ret.shape)
    np.save("../../../../extra/data/hrishikesh/labeled_points_image_1024_colored_"+type+".npy",ret[1:])
    return ret
