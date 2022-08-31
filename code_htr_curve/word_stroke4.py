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
from dataset import get_data
from transformer_stroke2 import Transformer
from model import strGen

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train(opt):
    """ dataset preparation """

    print("inside train")


    # loading data
    cp1_train = np.load('../../../extra/data/hrishikesh/indices_str_train.npy')[24:]
    cp2 = np.load('../../../extra/data/hrishikesh/keypoints_str_train.npy')
    cp2 = cp2.reshape((cp2.shape[0],21*3))
    cp2_train = torch.tensor(cp2)
    l_points = np.load('../../../extra/data/hrishikesh/labeled_point_stroke_sorted_train.npy')[1:,:]
    l_points = l_points.reshape((l_points.shape[0],1024,3))
    cp3_train = torch.tensor(l_points[:,:,:2])
    l_train = torch.tensor(l_points[:,:,2])
    #print(l_train)
    print('train data loaded'," ",cp2_train.size(),cp3_train.size(),l_train.size())

    type = 'val'
    if opt.val:
        type = 'test'

    cp1_val = np.load('../../../extra/data/hrishikesh/indices_str_'+type+'.npy')
    cp2 = np.load('../../../extra/data/hrishikesh/keypoints_str_'+type+'.npy')
    cp2 = cp2.reshape((cp2.shape[0],21*3))
    cp2_val = torch.tensor(cp2)
    l_points = np.load('../../../extra/data/hrishikesh/labeled_point_stroke_sorted_'+type+'.npy')
    l_points = l_points.reshape((l_points.shape[0],1024,3))[1:,:]
    cp3_val = torch.tensor(l_points[:,:,:2])
    l_val = torch.tensor(l_points[:,:,2])
    print('train data loaded'," ",cp2_val.size(),cp3_val.size(),l_val.size())

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
    #print("Model:")
    #print(model)
    #return

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
            points = cp3_train[i].float().to(device).unsqueeze(0)
            hand_track = cp2_train[int(data[0]):int(data[1]),:].float().to(device).unsqueeze(0)
            labels = l_train[i,:].long().to(device)
            #print(labels.shape)
            preds = model(hand_track,points).squeeze(0)
            #print(preds.size())
            cost = criterion(preds, labels)
            _,preds_str = preds.log_softmax(1).max(1)
            #print(preds_str.shape)
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

        cloud = np.zeros((1,1024*3))
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(cp1_val):
                points = cp3_val[i].float().to(device).unsqueeze(0)
                hand_track = cp2_val[int(data[0]):int(data[1]),:].float().to(device).unsqueeze(0)
                labels = l_val[i,:].long().to(device)
                preds = model(hand_track,points).squeeze(0)
                cost = criterion(preds, labels)
                print(i,' cost : ', cost )
                _,preds_str = preds.log_softmax(1).max(1)
                label_point_cloud = torch.cat((points,preds_str.unsqueeze(0).unsqueeze(2)),dim=2)
                #print(label_point_cloud.shape)
                cloud = np.append(cloud,label_point_cloud.detach().cpu().numpy().reshape((label_point_cloud.shape[0],1024*3)),axis=0)
                loss_avg_val.add(cost)


        prev_val = loss_avg_val.val()
        if min_val_loss > loss_avg_val.val():
            in_epoch = epoch
            min_val_loss = loss_avg_val.val()
            np.save("../../../extra/data/hrishikesh/label_point_cloud_transformer_MSE_adam_full_track_"+type+".npy",cloud[1:,:])
            print(cloud.shape)
            if not opt.val:
                torch.save(
                    model.state_dict(), f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}/best_model_MSE.pth')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--annFile', help='Where to store logs and models')
    parser.add_argument('--root', default= "../../../extra/data/hrishikesh/track/", help='Where to store logs and models')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
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
    parser.add_argument('--input_size', type=int, default=63, help='the size of the LSTM hidden state')
    parser.add_argument('--hidden_size', type=int, default=1024, help='the size of the LSTM hidden state')
    parser.add_argument('--output_size', type=int, default=2, help='number of iterations to train for')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(device)

    if not opt.exp_name:
        opt.exp_name = 'model_word_stroke_label_point_cloud_full_track_2'
        # print(opt.exp_name)


    #opt.saved_model = "../../../extra/data/hrishikesh/saved_models/model_word_stroke_label_point_cloud_full_track/best_model_MSE.pth"

    os.makedirs(f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}', exist_ok=True)

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    #random.seed(opt.manualSeed)
    #np.random.seed(opt.manualSeed)
    #torch.manual_seed(opt.manualSeed)
    #torch.cuda.manual_seed(opt.manualSeed)

    opt.num_gpu = torch.cuda.device_count()

    train(opt)
