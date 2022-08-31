from __future__ import print_function

import torch
#import torch.nn as nn
from torch.autograd import Variable

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import argparse
import time
import os
import pdb

from models import *
import synthTransformer as synthTrans
import hwroidataset as hwROIDat

parser = argparse.ArgumentParser(description='PyTorch HWNet Feature Extraction')

parser.add_argument('--img_folder', default='../../stroke/',help='image root folder')
parser.add_argument('--annFile', default='',help='test IAM file')
parser.add_argument('--exp_dir', default='output/', help='output directory to save files')
parser.add_argument('--type', default='val', help='output directory to save files')
parser.add_argument('--batch_size', type=int, default=6, help='batch_size')
parser.add_argument('--testAug',  action='store_true', default=False, help='perform test side augmentation')
parser.add_argument('--pretrained_file', default='pretrained/iam-model.t7', help='pre trained file path')
parser.add_argument('--exp_id', default='iam-test-0', help='experiment ID')

args = parser.parse_args()
print(args)

if args.annFile == '':
    args.annFile = '../ann/'+args.type+'_stroke_ann.txt'

save_dir = args.exp_dir + 'models/' + args.exp_id  + '/'
resume_dir = args.exp_dir + 'pretrained/' + args.exp_id  + '/'
logs_dir = args.exp_dir + 'logs/' + args.exp_id + '/'

if(not os.path.exists(resume_dir)):
    os.makedirs(resume_dir)

if(not os.path.exists(save_dir)):
    os.makedirs(save_dir)

if(not os.path.exists(logs_dir)):
    os.makedirs(logs_dir)

use_cuda = torch.cuda.is_available()

#Transformer
transform_test = transforms.Compose([
    synthTrans.Normalize(),
    synthTrans.ToTensor()
])

print('==> Resuming from checkpoint..')
checkpoint = torch.load(args.pretrained_file)
net = checkpoint['net']
net.eval()

#Scale to used for test time augmentation
if args.testAug:
    testFontSizes = [48,32,64]
else:
    testFontSizes = [48]

tCntr=0
for tSize in testFontSizes:
    #Dataset
    testset = hwROIDat.HWRoiDataset(ann_file=args.annFile,
                                        img_folder=args.img_folder,
                                        randFlag=False,
                                        valFlag = True,
                                        transform=transform_test,
                                        testFontSize=tSize)
    #Dataloader
    testloader = DataLoader(testset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    if tCntr==0:
        #feature matrix initialization
        featMat = np.zeros((len(testFontSizes),len(testset),2048))

    fCntr=0
    for batch_idx, data in enumerate(testloader):
        #print('batch idx: %d'%batch_idx)
        begin_time = time.time()
        inputs, targets, roi = data['image'], data['label'], data['roi']
        roi[:,0] = torch.arange(0,roi.size()[0])

        if use_cuda:
            inputs, targets, roi = inputs.cuda(), targets.cuda(), roi.cuda()
        inputs = inputs.unsqueeze(1)
        targets = targets.squeeze()

        inputs, targets, roi = Variable(inputs, volatile=True), Variable(targets), Variable(roi)

        outputs, outFeats = net(inputs, roi)
        featData = outFeats.cpu().data.numpy()

        #L2 Normalize of features
        normVal = np.sqrt(np.sum(featData**2,axis=1))
        featData = featData/normVal.reshape((targets.size(0),1))
        featMat[tCntr,fCntr:fCntr+targets.size(0),:] = featData
        fCntr+=targets.size(0)
        print('TSize:%d BatchIdx:%d/%d Epoch Time: %.4f secs' % (tSize, batch_idx,len(testloader),time.time()-begin_time))

    tCntr+=1

maxFeatMat = np.amax(featMat,axis=0)

# Save features
print('Saving features file')
np.save('../../feats_original_hwnet_stroke_'+args.type+'.npy',maxFeatMat)
print('Features file saved')
