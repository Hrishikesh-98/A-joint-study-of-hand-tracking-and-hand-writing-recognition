import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.init as init
import torchvision.transforms as transforms
from torch import optim
from utils import load_csv,load_npy, Averager, CTCLabelConverter, getVocab
from transformer2 import Transformer
import numpy as np
from Levenshtein import distance
import random, argparse, string, os
import json
from collections import OrderedDict
import gc
from dataset import get_data, get_points
from Cnn_Enc_Rnn_Dec import EncDec
from model import strGen, Model
from unet import UNet

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train(opt):
    """ dataset preparation """

    print("inside train")

    trainannFile = "hwnet/ann/test_new_ann.txt"
    valannFile = "hwnet/ann/val_new_ann.txt"
    testannFile = "hwnet/ann/train_stroke_ann.txt"

    # loading data
    if opt.val:
        trainannFile = "hwnet/ann/val_stroke_ann.txt"
        opt.root_img = "../../../extra/data/hrishikesh/stroke_black/"
        opt.root_mask = "../../../extra/data/hrishikesh/stroke_mask/"
    cp1_train = get_data(trainannFile, opt.root_img)
    l_train = get_data(trainannFile,opt.root_mask)
    #l_train = torch.tensor(l_points)
    cp1_train = torch.tensor(cp1_train)
    #l_train = torch.tensor(l_train)
    cp1_train = DataLoader(cp1_train, batch_size=opt.batch_size)
    print('train data loaded')

    if opt.val:
        cp1_val = get_data(testannFile, opt.root_img)
        l_val = get_data(testannFile,opt.root_mask)
    else:
        cp1_val = get_data(valannFile, opt.root_img)
        l_val = get_data(valannFile,opt.root_mask)
    print('val data loaded')
    cp1_val = torch.tensor(cp1_val)
    #l_val = torch.tensor(l_val)
    cp1_val = DataLoader(cp1_val, batch_size=opt.batch_size)
    model = UNet(n_channels=3, n_classes=2, bilinear=False)

    """ model configuration """

    model = model.to(device)
    model.train()

    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    #print(model)
    #return

    count = 0
    #for name, param in model.named_parameters():
    #    if 'FeatureExtraction' in name:
    #        param.requires_grad = False

    """ setup loss """
    criterion = nn.CrossEntropyLoss()
    loss_avg = Averager()
    loss_avg_val = Averager()
    min_val_loss = np.inf
    prev_t = 0
    prev_val =0

    # setup optimizer
    if opt.adam:
        #optimizer = optim.Adam(filtered_parameters, lr=opt.lr)
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    else:
        #optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
        optimizer = optim.RMSprop(model.parameters(), lr=opt.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    print("Optimizer:")
    print(optimizer)

    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    """ start training """
    in_epoch = 0

    for epoch in range(opt.num_iter):
        model.train()
        for i, data in enumerate(cp1_train):
            #break
            if opt.val:
                break
            image = data.to(device).float()
            true_masks = torch.tensor(l_train[i*opt.batch_size:i*opt.batch_size+opt.batch_size]).long().to(device)
            true_masks = (true_masks>0).long()
            preds = model(image)
            #print(preds.shape, ' ', true_masks.shape)
            #print(torch.max(true_masks,dim=1))
            cost = criterion(preds, true_masks)
            print(i," epoch : ", epoch, " val : ", min_val_loss,"in epoch : ",in_epoch," prev val : ",prev_val, " train : ", prev_t)
            print('cost : ', cost )

            model.zero_grad()
            cost.backward()
            optimizer.step()

            loss_avg.add(cost)
        if not opt.val:
            torch.save(
                model.state_dict(), f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}/iter_new_MSE.pth')

        prev_t = loss_avg.val()
        #if epoch < 150:
        #scheduler.step()
        #print(optimizer)

        cloud = np.zeros((1,256*256))
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(cp1_val):
                image = data.to(device).float()
                true_masks = torch.tensor(l_val[i*opt.batch_size:i*opt.batch_size+opt.batch_size]).long().to(device)
                true_masks = (true_masks>0).long()
                preds = model(image)
                #print(torch.max(preds_str,dim=1))
                cost = criterion(preds, true_masks)
                _,preds_str = preds.log_softmax(1).max(1)
                #print(preds)
                preds_str = preds_str
                #print(torch.max(preds_str,dim=1))
                #print(preds_str.shape)
                print(i,' cost : ', cost )
                if i < 101:
                    cloud = np.append(cloud,preds_str.detach().cpu().numpy().reshape((preds_str.shape[0],256*256))*255,axis=0)
                loss_avg_val.add(cost)


        prev_val = loss_avg_val.val()
        if min_val_loss > loss_avg_val.val():
            in_epoch = epoch
            min_val_loss = loss_avg_val.val()
            np.save("../../../extra/data/hrishikesh/image_train.npy",cloud)
            print(cloud.shape)
            if not opt.val:
                torch.save(
                    model.state_dict(), f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}/best_model_MSE.pth')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--annFile', help='Where to store logs and models')
    parser.add_argument('--root_img', default= "../../../extra/data/hrishikesh/words_dataset/", help='Where to store logs and models')
    parser.add_argument('--root_mask', default= "../../../extra/data/hrishikesh/words_dataset_mask/", help='Where to store logs and models')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--val', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=9, help='gradient clipping value. default=5')

    """ Model Architecture """
    parser.add_argument('--input_size', type=int, default=512, help='the size of the LSTM hidden state')
    parser.add_argument('--hidden_size', type=int, default=64, help='the size of the LSTM hidden state')
    parser.add_argument('--output_size', type=int, default=2, help='number of iterations to train for')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(device)

    if not opt.exp_name:
        opt.exp_name = 'model_iam_word_stroke_Resnet34_LSTMCell_stroke_Unet'
        # print(opt.exp_name)


    opt.saved_model = "../../../extra/data/hrishikesh/saved_models/model_iam_word_stroke_Resnet34_LSTMCell_stroke_Unet/best_model_MSE.pth"

    os.makedirs(f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}', exist_ok=True)

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    opt.num_gpu = torch.cuda.device_count()

    train(opt)
