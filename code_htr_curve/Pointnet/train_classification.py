"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np
import cv2 as cv
import datetime
import logging
import provider
import importlib
import shutil
import argparse
from collections import OrderedDict
import csv
import json

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=7003, type=int,  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1280, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--FT', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def load_model_json(model, path):
  data_dict = OrderedDict()
  with open(path, 'r') as f:
    data_dict = json.load(f)
  own_state = model.state_dict()
  for k, v in data_dict.items():
    print('Loading parameter:', k)
    if 'fc3' in k:
        continue
    if not k in own_state:
      print('Parameter', k, 'not found in own_state!!!')
      continue
    if type(v) == list or type(v) == int:
      v = torch.tensor(v)
    own_state[k].copy_(v)
  model.load_state_dict(own_state)
  print('Model loaded')

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, l_val,num_class):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    j = 0
    for points in tqdm(loader, total=len(loader)):

        target = l_val[j*args.batch_size:j*args.batch_size+args.batch_size]
        #points = cp_val[j]

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points.float())
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            #print(cat)
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
        j += 1

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc

def get_points_image(file):
    path = '../../../../extra/data/hrishikesh/words_dataset/'
    #path = '../../../../extra/data/hrishikesh/track/'
    im = cv.imread(path+file)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 190, 255, 0)
    invert = cv.bitwise_not(thresh)
    contours, hierarchy = cv.findContours(invert, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(invert, contours, -1, (255,255,255), 1)
    all_points = np.zeros((1,3))

    for h in range(invert.shape[0]):
        for w in range(invert.shape[1]):
            if invert[h][w] == 255:
                temp = np.zeros((1,3))
                temp[0][0] = w
                temp[0][1] = -h
                all_points = np.append(all_points,temp,axis=0)

    print(all_points[1:].shape)
    if all_points[1:].shape[0] < 128:
        l = np.zeros((128-all_points[1:].shape[0],3))
        for ls in l:
            ls[0] = all_points[-1][0]
            ls[1] = all_points[-1][1]
        all_points = np.append(all_points,l,axis=0)
    #np.save("./img.npy",all_points[1:])
    print(all_points[1:].shape)
    return torch.tensor(all_points[1:]).unsqueeze(0)

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('../../../../extra/data/hrishikesh/pointnet_pytorch/log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    cp = np.load('/net/voxel03/misc/extra/data/hrishikesh/labeled_points_image_1024_colored_train.npy')
    cp = cp.reshape((cp.shape[0],1024,6))
    #print(cp[0,0,:])
    l_train = np.load('/net/voxel03/misc/extra/data/hrishikesh/train_new_zero_labels_lower.npy')[:-1]
    cp = torch.tensor(cp)
    l_train = torch.tensor(l_train)
    trainDataLoader = torch.utils.data.DataLoader(cp, batch_size=args.batch_size)
    cp = zip(cp,l_train)
    # Create Dataloader of the above tensor with batch size
    trainDataLoaderZip = torch.utils.data.DataLoader(cp, batch_size=args.batch_size)
    #train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    cp = np.load('/net/voxel03/misc/extra/data/hrishikesh/labeled_points_image_1024_colored_val.npy')
    cp = cp.reshape((cp.shape[0],1024,6))
    l_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/val_new_zero_labels_lower.npy')
    cp = torch.tensor(cp)
    l_val = torch.tensor(l_val)
    testDataLoader = torch.utils.data.DataLoader(cp, batch_size=args.batch_size)
    cp = zip(cp,l_val)
    # Create Dataloader of the above tensor with batch size
    testDataLoaderZip = torch.utils.data.DataLoader(cp, batch_size=args.batch_size)
    #test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    #trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    #testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.FT:
        classifier = model.get_model(num_class, normal_channel=args.use_normals)
        print(classifier)
        classifier.apply(inplace_relu)
        load_model_json(classifier,'./best_model.json')
        #classifier.fc3 = nn.Linear(256,num_class)

        if not args.use_cpu:
            classifier = classifier.cuda()
            criterion = criterion.cuda()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        batch_id = 0
        for points in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
            #break
            optimizer.zero_grad()
            target = l_train[batch_id*args.batch_size:batch_id*args.batch_size+args.batch_size]
            #print(points.shape)
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            #print(points.shape)
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            batch_id += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, l_val,7003)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
