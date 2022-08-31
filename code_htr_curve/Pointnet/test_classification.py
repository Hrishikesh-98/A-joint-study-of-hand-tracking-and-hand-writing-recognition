"""
Author: Benny
Date: Nov 2019
"""
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from utils import load_csv
from collections import OrderedDict
import csv
import json
import cv2 as cv
from get_points import get_points
from utils import ColorTheArray

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def get_points_image(file):
    #path = '../../../../extra/data/hrishikesh/iiit-hws/'
    path = '../../../../extra/data/hrishikesh/words_dataset/'
    im = cv.imread(path+file)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 200, 255, 0)
    invert = cv.bitwise_not(thresh)
    contours, hierarchy = cv.findContours(invert, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(invert, contours, -1, (255,255,255), 1)
    all_points = np.zeros((1,6))

    for h in range(invert.shape[0]):
        for w in range(invert.shape[1]):
            if invert[h][w] == 255:
                temp = np.zeros((1,6))
                temp[0][0] = w
                temp[0][1] = h
                all_points = np.append(all_points,temp,axis=0)

    print(all_points[1:].shape)
    if all_points[1:].shape[0] < 128:
        l = np.zeros((128-all_points[1:].shape[0],6))
        for ls in l:
            ls[0] = all_points[-1][0]
            ls[1] = all_points[-1][1]
        all_points = np.append(all_points,l,axis=0)
    #np.save("./img.npy",all_points[1:])
    print(all_points[1:].shape)
    return torch.tensor(all_points[1:]).unsqueeze(0)


def load_model_json(model, path):
  data_dict = OrderedDict()
  with open(path, 'r') as f:
    data_dict = json.load(f)    
  own_state = model.state_dict()
  for k, v in data_dict.items():
    print('Loading parameter:', k)
    if not k in own_state:
      print('Parameter', k, 'not found in own_state!!!')
    if type(v) == list or type(v) == int:
      v = torch.tensor(v)
    own_state[k].copy_(v)
  model.load_state_dict(own_state)
  print('Model loaded')


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def test(model, loader):
    classifier = model.eval()
    y = np.zeros((1,128*640))
    xyz = np.zeros((1,128*3))
    #f = open("../../dataset_val2.txt","r")
    #lines = f.readlines()
    for i,points in enumerate(loader):
    #for i,line in enumerate(lines):
        print(i)
        #values = line.split()[1]
        #try:
            #points = get_points('npzs_test','dataset_test',i,1)
            #points = torch.tensor(points).unsqueeze(0)
        if not args.use_cpu:
            points = points.float().cuda()
        else:
            points = points.float()
        points = points.transpose(2, 1)
        x, xyz2 = classifier(points)
        #print(x.shape)
        #except:
            #print("except")
            #points = get_points('npzs_test','test_ann',i,1)
            #points = torch.tensor(points).unsqueeze(0)
            #if not args.use_cpu:
            #    points = points.float().cuda()
            #else:
            #    points = points.float()
            #points = points.transpose(2, 1)
            #x = classifier(points)
            #print(x.shape)
        xyz2 = xyz2.permute(0,2,1)
        x = x.permute(0,2,1)
        xyz2 = torch.reshape(xyz2,(xyz2.shape[0],128*3))
        x = torch.reshape(x,(x.shape[0],128*640))
        #y = np.append(y,x.cpu().reshape((x.shape[0],128*640)),axis=0)
        y = np.append(y,x.detach().cpu().numpy(),axis=0)
        #y = torch.cat(y,x,dim=0)
        xyz2 = xyz2.detach().cpu().numpy()
        #xyz2 = xyz2.reshape((xyz2.shape[0],196*3))
        #xyz2 = xyz2[np.argsort(xyz2[:,0])]
        #xyz2 = xyz2.reshape((1,128*3))
        xyz = np.append(xyz,xyz2,axis=0)
    return y ,xyz


def samplePointcloud(model):
    classifier = model.eval()
    y = np.zeros((1,128*640))
    xyz = np.zeros((1,1024*3))
    f = open("../hwnet/ann/test_new_ann.txt","r")
    lines = f.readlines()
    print(len(lines))
    for i,line in enumerate(lines): #[100000:150000]):
        print(i)
        values = line.split()[0]
        points = get_points_image(values)
        if not args.use_cpu:
            points = points.float().cuda()
        else:
            points = points.float()
        points = points.transpose(2, 1)
        x, xyz2 = classifier(points)
        xyz2 = xyz2.permute(0,2,1).squeeze(0)
        xyz2 = xyz2.detach().cpu().numpy()
        print(xyz2.shape)
        xyz2 = xyz2[np.argsort(xyz2[:,0])]
        xyz2 = xyz2.reshape((1,1024*3))
        xyz = np.append(xyz,xyz2,axis=0)
        #y = np.append(y,x.cpu().reshape((x.shape[0],128*640)),axis=0)
        #y = np.append(y,x.cpu(),axis=0)
    return y, xyz




def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    #experiment_dir = 'log/classification/' + args.log_dir
    if args.log_dir == "pointnet2_msg_normals":
        experiment_dir = 'log/classification/' + args.log_dir
        #classifier = model.get_model(num_class,args.use_normals)
    elif args.log_dir == "pointnet2_ssg_wo_normals":
        experiment_dir = 'log/classification/' + args.log_dir
        #classifier = model.get_model(num_class,args.use_normals)
    elif  args.log_dir == "pointnet2_part_seg_msg":
        experiment_dir = 'log/part_seg/' + args.log_dir
        #classifier = model.get_model(50,args.use_normals)
    else:
        experiment_dir = 'log/sem_seg/' + args.log_dir
        #classifier = model.get_model(13)


    '''LOG'''
    args = parse_args()

    '''DATA LOADING'''
    cp= np.load('/net/voxel03/misc/extra/data/hrishikesh/labeled_points_image_1024_colored_train.npy')
    cp = cp.reshape((cp.shape[0],1024,6))
    print('train data loaded',cp.shape)

    #Create Dataloader of the above tensor with batch size
    cp_train = DataLoader(cp, batch_size=args.batch_size)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    print(model_name)
    model = importlib.import_module(model_name)

    if args.log_dir == "pointnet2_msg_normals":
        experiment_dir = 'log/classification/' + args.log_dir
        classifier = model.get_model(num_class,args.use_normals)
    elif args.log_dir == "pointnet2_ssg_wo_normals":
        experiment_dir = 'log/classification/' + args.log_dir
        classifier = model.get_model(num_class,args.use_normals)
    elif  args.log_dir == "pointnet2_part_seg_msg":
        experiment_dir = 'log/part_seg/' + args.log_dir
        classifier = model.get_model(50,args.use_normals)
    else:
        experiment_dir = 'log/sem_seg/' + args.log_dir
        classifier = model.get_model(13)

    if not args.use_cpu:
        classifier = classifier.cuda()

    if args.log_dir == "pointnet2_msg_normals":
        print('in')
        #load_model_json(classifier,'./best_model.json')
    elif args.log_dir == "pointnet2_ssg_wo_normals":
        load_model_json(classifier,'./best_model_ssg.json')
    elif  args.log_dir == "pointnet2_part_seg_msg":
        load_model_json(classifier,'./best_model_partseg.json')
    else:
        load_model_json(classifier,'./best_sem_seg.json')

    print(classifier)
    torch.save(
        classifier.state_dict(), f'../../../../extra/data/hrishikesh/best_original_pointnet.pth')
    with torch.no_grad():
        vec,points = test(classifier.eval(), cp_train)
        #vec,points = samplePointcloud(classifier.eval())

    #print(points.shape)

    v = vec[1:]
    p = points[1:]
    print(p.shape, ' ', v.shape)

    val = "train2" #train4
    
    np.save('/net/voxel03/misc/extra/data/hrishikesh/group_rep_labeled_128_'+val+'.npy',v)
    np.save('/net/voxel03/misc/extra/data/hrishikesh/labeled_points_image_128_'+val+'.npy',p)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
