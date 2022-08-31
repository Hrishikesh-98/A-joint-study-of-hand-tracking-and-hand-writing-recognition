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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train(opt):
    """ dataset preparation """

    print("inside train")


    # loading data
    cp1_train = np.load('../../../extra/data/hrishikesh/indices.npy')[0:20]
    #temp = np.load('/net/voxel03/misc/extra/data/hrishikesh/indices_all.npy')[25:37]
    #cp1_train = np.append(cp1_train,temp,axis=0)
    cp2 = np.load('/net/voxel03/misc/extra/data/hrishikesh/feature_image.npy')
    cp3 = np.load('/net/voxel03/misc/extra/data/hrishikesh/group_rep_new_track_black.npy')
    cp3 = np.append(cp3,np.zeros((5483-cp3.shape[0],128*640)),axis=0)
    cp3 = cp3.reshape((cp3.shape[0],128,640))
    l_train = np.load('/net/voxel03/misc/extra/data/hrishikesh/keypoints.npy')
    cp1_train = torch.tensor(cp1_train)
    cp2 = torch.tensor(cp2)
    cp3 = torch.tensor(cp3)
    l_train = torch.tensor(l_train)
    print('train data loaded'," ",cp1_train.size(),cp2.size(),cp3.size(),l_train.shape)

    if opt.val:
        cp1_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/indices.npy')[0:25]
        print(cp1_val)
    else:
        cp1_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/indices.npy')[20:25]
        #temp = np.load('/net/voxel03/misc/extra/data/hrishikesh/indices_all.npy')[37:]
        #print(temp.shape)
        #cp1_val = np.append(cp1_val,temp,axis=0)
    cp2 = np.load('/net/voxel03/misc/extra/data/hrishikesh/feature_image.npy')   
    cp3 = np.load('/net/voxel03/misc/extra/data/hrishikesh/group_rep_new_track_black.npy')
    cp3 = np.append(cp3,np.zeros((5483-cp3.shape[0],128*640)),axis=0)
    cp3 = cp3.reshape((cp3.shape[0],128,640))
    l_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/keypoints.npy')
    cp1_val = torch.tensor(cp1_val)
    cp2 = torch.tensor(cp2)
    cp3 = torch.tensor(cp3)
    l_val = torch.tensor(l_val)
    print('val data loaded'," ",cp1_val.size(),cp2.size(),cp3.size(),l_train.shape)

    model = Transformer()

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
    print(model)
    #return

    """ setup loss """
    criterion = nn.MSELoss()
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
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    """ start training """
    in_epoch = 0

    for epoch in range(opt.num_iter):
        model.train()
        for i, data in enumerate(cp1_train):
            if opt.val:
                break
            #src = cp3[i,:,:].to(device).float().unsqueeze(0)
            tgt = cp2[int(data[0]):int(data[1]),:].to(device).float().unsqueeze(0)
            preds = model(tgt)
            preds = torch.reshape(preds,((preds.shape[1],21,3)))
            labels = l_train[int(data[0]):int(data[1]),:].float().unsqueeze(0)
            labels = torch.reshape(labels,((labels.shape[1],21,3)))
            cost = criterion(preds, labels.cuda())
            print(i," epoch : ", epoch, " val : ", min_val_loss,"in epoch : ",in_epoch," prev val : ",prev_val, " train : ", prev_t)
            print('cost : ', cost )

            model.zero_grad()
            cost.backward()
            optimizer.step()

            loss_avg.add(cost)
        if not opt.val:
            torch.save(
                model.state_dict(), f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}/iter_new_MSE_enc.pth')

        prev_t = loss_avg.val()
        #if epoch < 150:
        #scheduler.step()
        print(optimizer)

        keys = np.zeros((1,63))
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(cp1_val):
                #src = cp3[i,:,:].to(device).float().unsqueeze(0)
                tgt = cp2[int(data[0]):int(data[1]),:].to(device).float().unsqueeze(0)
                preds = model(tgt)
                preds = torch.reshape(preds,((preds.shape[1],21,3)))
                labels = l_train[int(data[0]):int(data[1]),:].float().unsqueeze(0)
                labels = torch.reshape(labels,((labels.shape[1],21,3)))
                cost = criterion(preds, labels.cuda())
                print(i,' cost : ', cost )
                keys = np.append(keys,preds.detach().cpu().numpy().reshape((preds.shape[0],63)),axis=0)


                loss_avg_val.add(cost)


        prev_val = loss_avg_val.val()
        if min_val_loss > loss_avg_val.val():
            in_epoch = epoch
            min_val_loss = loss_avg_val.val()
            np.save("../../../extra/data/hrishikesh/keys_MSE_adam_enc.npy",keys)
            print(keys.shape)
            if not opt.val:
                torch.save(
                    model.state_dict(), f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}/best_model_MSE_enc.pth')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
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
    parser.add_argument('--hidden_size', type=int, default=128, help='the size of the LSTM hidden state')
    parser.add_argument('--out', nargs="+", default=[128,256], help='number of iterations to train for')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(device)

    if not opt.exp_name:
        opt.exp_name = 'model_hand_tracking'
        # print(opt.exp_name)


    opt.saved_model = "../../../extra/data/hrishikesh/saved_models/model_hand_tracking/best_model_MSE_enc.pth"

    os.makedirs(f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}', exist_ok=True)

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    opt.num_gpu = torch.cuda.device_count()

    train(opt)
