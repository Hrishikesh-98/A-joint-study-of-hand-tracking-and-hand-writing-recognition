import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.nn.init as init
from torch import optim
from model import Model, LSTM, FFN, ILSTM, PointHWnet, Mdl
import numpy as np
import pandas as pd
import csv
import random
from scipy import spatial
import argparse
import os
import string
from utils import Averager, CTCLabelConverter, load_csv, AttnLabelConverter
from Levenshtein import distance
import fastwer
from spellchecker import SpellChecker

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def val(opt):

    """ dataset preparation """
    # loading data
    spell = SpellChecker()

    cp1 = np.load('../../../extra/data/hrishikesh/feats_original_hwnet_'+opt.type+'.npy')
    if opt.model_type != 6:
      cp2 = np.load('/net/voxel03/misc/extra/data/hrishikesh/semantic_rep_new_'+opt.type+'.npy')
      #cp2 = cp2[:cp1.shape[0]]
      cp = np.hstack((cp1,cp2))
    else:
      cp = cp1
    l_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/text_new_'+opt.type+'.npy')
    cp1 = torch.tensor(cp1)

    if opt.model_type == 6:
      cp2 = np.load('/net/voxel03/misc/extra/data/hrishikesh/semantic_rep_new_'+opt.type+'.npy')
      #cp2 = np.load('../../../extra/data/hrishikesh/points_new_'+opt.type+'.npy')
      cp2 = torch.tensor(cp2)
      #cp2 = torch.reshape(cp2,(cp2.size()[0],1280,6))
      print(cp2.size())


    #if opt.model_type != 3 and opt.model_type != 4 and opt.model != 6:
    #  cp = torch.reshape(cp, (cp.size()[0],32,96))
    print('train data loaded',cp1.size())
    
    print(opt.type+' data loaded', cp2.size())

    print(l_val.shape)
    print(l_val)


    # Create Dataloader of the above tensor with batch size
    cp_val = DataLoader(cp, batch_size=opt.batch_size)
    if opt.model_type == 6:
      cp_val = DataLoader(cp2, batch_size=opt.batch_size)
      cp_val_hw = DataLoader(cp1, batch_size=opt.batch_size)
    
    """ model configuration """
    
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)

    opt.num_class = len(converter.character)

    if opt.model_type == 1:
      model = Model(opt)

    elif opt.model_type == 2:
      model = LSTM(opt)

    elif opt.model_type == 4:
      model = ILSTM(opt)
    
    elif opt.model_type == 6:
      model = Mdl(opt)

    else:
      model = FFN(opt)

    model = model.to(device)
    print(model)

    model.load_state_dict(torch.load('../../../extra/data/hrishikesh/saved_models/model_hwnet_original_text/'+opt.model))
    #model.load_state_dict(torch.load('./saved_models/model_PointnetFFN5/'+opt.model))
    
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

    
    loss_avg = Averager()

    count  = 0
    dist  = 0
    find  = list()
    total = list()

    for i in range(len(opt.find)):
      find.append(0)
      total.append(0)

    model.eval()

    hypo = []
    output = []
    if opt.model_type == 6:
      data_hw = iter(cp_val_hw)
    for i, data in enumerate(cp_val):
        # val part
        image_tensors = data
        labels = l_val[i*opt.batch_size:i*opt.batch_size+opt.batch_size]
        image = image_tensors.float().to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        # batch_size = image.size(0)
        if opt.model_type == 6:
           #image = image.transpose(2,1)
           hw = next(data_hw).float().to(device)
        #   preds = model(image,hw.float().to(device))
        # else:
        if 'CTC' in opt.Prediction:
            preds = model(image,hw)
            preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)

            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
            print(preds_str[:5], " ", labels[: 5])

            preds = preds.log_softmax(2).permute(1, 0, 2)
            cost = criterion(preds, text, preds_size, length)

        else:
            preds = model(image,text[:,:-1])
            target = text[:,:-1]
            preds_size = torch.IntTensor([preds.size(1)] * opt.batch_size)

            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
            print(preds_str[:5], " ", labels[: 5])

            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        print("cost : ", cost)

        loss_avg.add(cost)

        if opt.words:
            for i in range(len(labels)):
                if opt.spell_check:
                    if not opt.case_sensitive:
                        b = spell.correction(preds_str[i].lower())
                    else:
                        b = spell.correction(preds_str[i])
                    b1 = spell.correction(preds_str[i])
                else:
                    b = preds_str[i]
                    b1 = preds_str[i]
                dist += distance(b,labels[i])
                hypo.append(labels[i].lower())
                output.append(b)
                for j in range(len(opt.find)):
                  if b1 == opt.find[j] and labels[i] == opt.find[j]:
                    find[j] += 1
                  if labels[i] == opt.find[j]:
                    total[j] += 1
                if b == labels[i].lower():
                    count += 1

    print('average loss : ', loss_avg.val())
    if opt.words:
        avg_dist = dist/len(l_val)
        print("Correct Words Predicted : ", count,'/',len(l_val))
        print('Avg Levenshtein Distance : ',avg_dist)
        for i in range(len(opt.find)):
          print(opt.find[i], " : ", find[i],"/",total[i])
        print(sum(find), "/", sum(total))
        print("accuracy : ",count/len(l_val))

    # Corpus-Level WER: 40.0
    print(fastwer.score(hypo, output))
    # Corpus-Level CER: 25.5814
    print(fastwer.score(hypo, output, char_level=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--batch_size', type=int, default=20, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--model', type=str, default='iter_100.pth', help='number of iterations to train for')
    parser.add_argument('--type', type=str, default='val', help='number of iterations to train for')
    parser.add_argument('--words', type=bool, default=False, help='number of iterations to train for')
    parser.add_argument('--find', nargs="+", default=["a", "b"], help='number of iterations to train for')
    parser.add_argument('--spell_check', action="store_true", help='number of iterations to train for')
    parser.add_argument('--case_sensitive',action="store_true", help='number of iterations to train for')
    parser.add_argument('--model_type', type=int, default=1, help='for random seed setting')

    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz\'/\\.,?"(){}[]-_=+@!$%:#&*;', help='character label')
    parser.add_argument('--Prediction', type=str, default='CTC', help='gradient clipping value. default=5')
    """ Model Architecture """
    parser.add_argument('--input_size', type=int, default=96, help='the size of the LSTM hidden state')
    parser.add_argument('--hidden_size', type=int, default=128, help='the size of the LSTM hidden state')
    parser.add_argument('--out', nargs="+", default=[2048,1024,512,256,128,50], help='number of iterations to train for')
    opt = parser.parse_args()

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    opt.num_gpu = torch.cuda.device_count()
    opt.character = string.printable[:-6]


    device = 'cuda'

    val(opt)
