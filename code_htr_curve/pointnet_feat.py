import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.init as init
from model import FFN, Model, CLSTM, PointHWnet
from hwnet.pytorch import synthTransformer as synthTrans
from hwnet.pytorch import hwroidataset as hwROIDat
import torchvision.transforms as transforms
from torch import optim
from utils import load_csv,load_npy, Averager, CTCLabelConverter
#from Pointnet.models.pointnet2_cls_msg_ import Pointnet2
#from Pointnet.models.pointnet2_cls_ssg import Pointnet
from model import Mdl16 as Mdl
import numpy as np
from Levenshtein import distance
import random, argparse, string, os
import json
from collections import OrderedDict
import gc

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def load_model_json(model,str, path, i = 0):
  data_dict = OrderedDict()
  with open(path, 'r') as f:
    data_dict = json.load(f)
  own_state = model.state_dict()
  count = 0
  for k in own_state:
    print(count, " ",k)
    count +=1
  for k, v in data_dict.items():
    if i == 1:
      k = k[7:]
    else:
      k = str+k
    print('Loading parameter:', k)
    if not k in own_state:
      print('Parameter', k, 'not found in own_state!!!')
      continue
    if type(v) == list or type(v) == int:
      v = torch.tensor(v)
    own_state[k].copy_(v)
    own_state[k].requires_grad = False
  model.load_state_dict(own_state)
  print('Model loaded')


def get_data(model,cp1_val,cp2_val,type):
    output = np.zeros((1,26*95))
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(cp2_val):
            image_tensors = data
            data_hw = cp1_val[i]
            image = image_tensors.float().to(device)
            batch_size = image.size(0)
            image = image.transpose(2,1)
            preds,out = model(image, data_hw)
            output = np.append(output,out.reshape((out.shape[0],out.shape[1]*out.shape[2]),axis=0))

            print(output.shape)
    np.save('/net/voxel03/misc/extra/data/hrishikesh/points_cumalative_'+type+'.npy',output[1:])


def train(opt):
    """ dataset preparation """

    print("inside train")

    types = opt.Modeltype.split("_")
    print(types)
    cp1 = np.load('/net/voxel03/misc/extra/data/hrishikesh/feats_original_hwnet_'+opt.type+'.npy')
    cp2 = np.load('/net/voxel03/misc/extra/data/hrishikesh/points_new_'+opt.type+'.npy')
    cp2 = cp2.reshape((cp2.shape[0],1280,6))
    if "512" in types and "image" in types:
        cp2 = np.load('/net/voxel03/misc/extra/data/hrishikesh/sample_points_image_'+opt.type+'.npy')
        cp2 = cp2.reshape((cp2.shape[0],512,3))
    if "1024" in types and "image" in types:
        cp2 = np.load('/net/voxel03/misc/extra/data/hrishikesh/labeled_points_image_1024_'+opt.type+'.npy')
        cp2 = cp2.reshape((cp2.shape[0],1024,4))
    cp3 = np.load('/net/voxel03/misc/extra/data/hrishikesh/points_new_'+opt.type+'.npy')
    cp3 = cp3.reshape((cp3.shape[0],1280,6))
    cp3 = torch.tensor(cp3)
    cp3_val = DataLoader(cp3, batch_size=opt.batch_size)

    if "128" in types:
        l_vecs = np.load('../../../extra/data/hrishikesh/group_rep_image_'+opt.type+'.npy')
        l_vecs = l_vecs.reshape((l_vecs.shape[0],128,640))
        l_vecs = torch.tensor(l_vecs)
        print("1")
        l_points = np.load('../../../extra/data/hrishikesh/sample_points_image_128_'+opt.type+'.npy')
        l_points = l_points.reshape((l_points.shape[0],128,3))
        l_points = torch.tensor(l_points)
        print("2")
        cp2 = torch.cat((l_points,l_vecs),dim=2)
        print("3")
    l_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/'+opt.type+'_new_labels.npy')
    cp1 = torch.tensor(cp1)
    cp2 = torch.tensor(cp2)
    print('val data loaded ', cp1.shape, cp2.shape)


    # Create Dataloader of the above tensor with batch size
    cp1_val = DataLoader(cp1, batch_size=opt.batch_size)
    cp2_val = DataLoader(cp2, batch_size=opt.batch_size)

    feats_val = []


    """ model configuration """
    model = Mdl(opt)


    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    model = model.to(device)

    if opt.saved_model != '':
            model.load_state_dict(torch.load(opt.saved_model))
            print(opt.saved_model)

    """ start training """
    feats = np.zeros((1,2048))
    model.eval()
    data_hw = iter(cp1_val)
    data_points = iter(cp3_val)
    with torch.no_grad():
        for i, data in enumerate(cp2_val):
            image_tensors = data
            image = image_tensors.float().to(device)
            image = image.transpose(2,1)
            if "1024" in types or "1280" in types:
                preds, OutFeats = model(image,next(data_points).float().to(device).transpose(2,1), next(data_hw).float().to(device))
            else:
                preds, OutFeats = model(image, next(data_hw).float().to(device))
            feats = np.append(feats,OutFeats.cpu().numpy(),axis=0)
            print(i,' ', feats[1:].shape)
            
    #L2 Normalize of features
    feats = feats[1:]
    normVal = np.sqrt(np.sum(feats**2,axis=1))
    feats = feats/normVal.reshape((feats.shape[0],1))
    # Save features
    save_dir = "../../../extra/data/hrishikesh/" + opt.exp_id + "/"
    print('Saving features file ',save_dir)
    if opt.pointnet:
        np.save(save_dir+'feats_pointnet_'+opt.type+opt.Modeltype+opt.Savetype+'.npy',feats)
    else:
        np.save(save_dir+'feats_'+opt.type+opt.Modeltype+opt.Savetype+'.npy',feats)
    print('Features file saved ',feats.shape)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--batch_size', type=int, default=48, help='input batch size')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--type', default='train')
    parser.add_argument('--Savetype', default='4000')
    parser.add_argument('--Modeltype', default='128_1024')
    parser.add_argument('--pointnet',  action='store_true', default=False, help='to print out list in text files')

    """ HwNet """
    parser.add_argument('--exp_dir', default='output/')
    parser.add_argument('--exp_id', default='iam-test-0', help='experiment ID')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    if opt.pointnet:
        opt.saved_model = "../../../extra/data/hrishikesh/saved_models/model_pointnet_label"+opt.Modeltype+"/best_model_both_"+opt.Savetype+".pth"

    else:
        opt.saved_model = "../../../extra/data/hrishikesh/saved_models/model_hwpoint_original_label"+opt.Modeltype+"/best_model_both_"+opt.Savetype+".pth"

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    opt.num_gpu = torch.cuda.device_count()

    train(opt)
