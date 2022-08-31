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
from transformer_stroke import Transformer
from model import strGen

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train(opt):
    """ dataset preparation """

    print("inside train")


    # loading data
    images = get_data("./hwnet/ann/track_new_ann.txt","../../../extra/data/hrishikesh/track/")
    cp1_train = np.load('../../../extra/data/hrishikesh/indices.npy')[0:20]
    #cp2_train = torch.tensor(np.load('../../../extra/data/hrishikesh/feature_image_196.npy'))[:20]
    #cp2_train = cp2_train.reshape((cp2_train.shape[0],196,512))
    cp2_train = images[:20]
    cp3 = np.load('../../../extra/data/hrishikesh/keypoints.npy')
    cp3 = cp3.reshape((cp3.shape[0],21*3))
    cp1_train = torch.tensor(cp1_train)
    cp3 = torch.tensor(cp3)
    l_points = np.load('../../../extra/data/hrishikesh/sample_points_track.npy')
    l_points = l_points.reshape((l_points.shape[0],128,3))[:,:,:2]
    #print(l_points)
    #temp = l_points.reshape((l_points.shape[0]*196,2))
    #out, inds = torch.max(torch.tensor(temp),dim=0)
    #print(out, inds)
    #out, inds = torch.min(torch.tensor(temp),dim=0)
    #print(out, inds)
    #l_points = l_points.reshape((l_points.shape[0],128,2))
    l_train = torch.tensor(l_points)
    b = torch.tensor([ 162, -61])
    l_train = torch.div(l_train,b)
    #print(l_train)
    #l_train = torch.reshape((l_points,l_vecs),dim=2)
    print('train data loaded'," ",cp1_train.size(),cp3.size(),l_train.shape)

    if opt.val:
        cp1_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/indices.npy')[0:25]
        #cp2_val = torch.tensor(np.load('../../../extra/data/hrishikesh/feature_image_196.npy'))[0:25]
        #cp2_val = cp2_val.reshape((cp2_val.shape[0],196,512))
        cp2_val = images
        print(cp1_val)
    else:
        cp1_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/indices.npy')[20:25]
        #cp2_val = torch.tensor(np.load('../../../extra/data/hrishikesh/feature_image_196.npy'))[20:25]
        #cp2_val = cp2_val.reshape((cp2_val.shape[0],196,512))
        cp2_val = images[20:25]
    cp1_val = torch.tensor(cp1_val)
    #cp2_val = torch.tensor(cp2_val)
    print('val data loaded'," ",cp1_val.size(),cp3.size(),l_train.shape)

    model = strGen(opt)

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
            image = cp2_train[i].to(device).float()
            l =  int(data[1])-int(data[0])
            #if int(data[1])-int(data[0]) < 33:
            #    temp = torch.empty((33-l, 21*3))
            #    for j in range(33-l):
            #        temp[j,:] = cp3[int(data[1]),:]
                #print(temp)
            #    hand_track = torch.cat((cp3[int(data[0]):int(data[1])],temp),dim=0).to(device).float().unsqueeze(0)
            #else:
            hand_track = torch.empty((l, 21*3))
            temp = cp3[int(data[0]):int(data[1]),:]
            #print(temp)
            for k in range(l):
                for j in range(21):
                    hand_track[k,j*3] = temp[k,j*3]-temp[0,0]
                    hand_track[k,j*3+1] = temp[k,j*3+1]-temp[0,1]
                    hand_track[k,j*3+2] = temp[k,j*3+2]
                    #print(j*3, ' previous val ', temp[k,j*3], ' subtracting ', temp[0,0] ,' current_val ', hand_track[k,j*3])
                    #print(j*3+1, ' previous val ', temp[k,j*3+1], ' subtracting ', temp[0,0] ,' current_val ', hand_track[k,j*3+1])
                #hand_track = cp3[int(data[0]):int(data[1]),:].to(device).float().unsqueeze(0)
            #print(l, ' ', hand_track)
            hand_track = hand_track.to(device).float().unsqueeze(0)
            point_cloud = l_train[i,:].float().to(device).unsqueeze(0)
            preds = model(hand_track,image)
            cost = criterion(preds, point_cloud)
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

        cloud = np.zeros((1,128*2))
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(cp1_val):
                image = cp2_val[i].to(device).float()
                l =  int(data[1])-int(data[0])
                #if int(data[1])-int(data[0]) < 33:
                #    temp = torch.empty((33-l, 21*3))
                #    for j in range(33-l):
                #        temp[j,:] = cp3[int(data[1]),:]
                #    hand_track = torch.cat((cp3[int(data[0]):int(data[1])],temp),dim=0).to(device).float().unsqueeze(0)
                #else:
                #    hand_track = cp3[int(data[0]):int(data[1]),:].to(device).float().unsqueeze(0)
                hand_track = torch.empty((l, 21*3))
                temp = cp3[int(data[0]):int(data[1]),:]
                #print(temp)
                for k in range(l):
                    for j in range(21):
                        hand_track[k,j*3] = temp[k,j*3]-temp[0,0]
                        hand_track[k,j*3+1] = temp[k,j*3+1]-temp[0,1]
                        hand_track[k,j*3+2] = temp[k,j*3+2]
                        #print(j*3, ' previous val ', temp[k,j*3], ' subtracting ', temp[0,0] ,' current_val ', hand_track[k,j*3])
                        #print(j*3+1, ' previous val ', temp[k,j*3+1], ' subtracting ', temp[0,0] ,' current_val ', hand_track[k,j*3+1])
                    #hand_track = cp3[int(data[0]):int(data[1]),:].to(device).float().unsqueeze(0)
                #print(l, ' ', hand_track)
                hand_track = hand_track.to(device).float().unsqueeze(0)
                point_cloud = l_train[i,:].float().to(device).unsqueeze(0)
                preds = model(hand_track,image)
                cost = criterion(preds, point_cloud)
                print(i,' cost : ', cost )
                cloud = np.append(cloud,preds.detach().cpu().numpy().reshape((preds.shape[0],128*2)),axis=0)
                loss_avg_val.add(cost)


        prev_val = loss_avg_val.val()
        if min_val_loss > loss_avg_val.val():
            in_epoch = epoch
            min_val_loss = loss_avg_val.val()
            np.save("../../../extra/data/hrishikesh/cloud_transformer_MSE_adam.npy",cloud)
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
        opt.exp_name = 'model_word_stroke_VGG_LStm'
        # print(opt.exp_name)


    opt.saved_model = "../../../extra/data/hrishikesh/saved_models/model_word_stroke_VGG_LStm/best_model_MSE.pth"

    os.makedirs(f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}', exist_ok=True)

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    #random.seed(opt.manualSeed)
    #np.random.seed(opt.manualSeed)
    #torch.manual_seed(opt.manualSeed)
    #torch.cuda.manual_seed(opt.manualSeed)

    opt.num_gpu = torch.cuda.device_count()

    train(opt)
