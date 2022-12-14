''' Created on Dec 26, 2017
@author: praveen krishnan
'''

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.neighbors import KDTree
import os
import argparse

import pdb


parser = argparse.ArgumentParser(description='PyTorch HWNet Evaluation for Word Spotting')

parser.add_argument('--Type', default='val',help='test annotation file')
parser.add_argument('--Loadtype', default='val',help='test annotation file')
parser.add_argument('--Modeltype', default='',help='test annotation file')
parser.add_argument('--stopword_file', default='../ann/stopwords.txt',help='test query file')
parser.add_argument('--exp_dir', default='../../../extra/data/hrishikesh/', help='root exp folder')
parser.add_argument('--exp_id', default='iam-test-0', help='exp id')
parser.add_argument('--printFlag',  action='store_true', default=False, help='to print out list in text files')
parser.add_argument('--removeQuery',  action='store_true', default=False, help='seperate out query from candidate list. Only for Botany/Konz datasets')
parser.add_argument('--outSize', type=int, default=500, help='batch_size')
parser.add_argument('--hwnet',  action='store_true', default=False, help='to print out list in text files')
parser.add_argument('--pointnet',  action='store_true', default=False, help='to print out list in text files')
parser.add_argument('--og_pointnet',  action='store_true', default=False, help='to print out list in text files')
parser.add_argument('--noStopwords',  action='store_true', default=False, help='to print out list in text files')

args = parser.parse_args()

annFile = 'hwnet/ann/' + args.Type + '_new_ann.txt'
if args.noStopwords:
    queryFile = 'hwnet/ann/query_no_stopwords_' + args.Type +'.txt'
else:
    queryFile = 'hwnet/ann/query_' + args.Type +'.txt'

featFolder = os.path.join(args.exp_dir, args.exp_id)
logs_dir = os.path.join(args.exp_dir,'logs',args.exp_id)
ret_dir = os.path.join(args.exp_dir,'retrieval',args.exp_id)

if(not os.path.exists(featFolder)):
    os.makedirs(featFolder)
if(not os.path.exists(logs_dir)):
    os.makedirs(logs_dir)
if(not os.path.exists(ret_dir)):
    os.makedirs(ret_dir)
if(not os.path.exists(os.path.join(ret_dir,'OutList'))):
    os.makedirs(os.path.join(ret_dir,'OutList'))


#Test time augmentation
if args.og_pointnet:
    featMat = np.load('../../../extra/data/hrishikesh/semantic_rep_image_'+ args.Type+'.npy')
    print("OG pointnet")
elif args.pointnet:
    featMat = np.load(os.path.join(featFolder,'feats_pointnet_'+ args.Type+ args.Modeltype +args.Loadtype+'.npy'))
    print("pointnet")
elif not args.hwnet:
    featMat = np.load(os.path.join(featFolder,'feats_'+ args.Type+ args.Modeltype +args.Loadtype+'.npy'))
else:
    featMat = np.load("../../../extra/data/hrishikesh/feats_original_hwnet_"+args.Type+'.npy')
    print("hwnet")

vocab = {}
vocabIdx = {}
vocabCntr = {}
vCntr=0
labels = []
wordPaths = []

with open(annFile) as aFile:
    for line in aFile:
        tempStr = line.split()
        wordPaths.append(tempStr[0])
        if(tempStr[1] in vocab):
            labels.append(vocab[tempStr[1]])
            vocabCntr[vocab[tempStr[1]]] = vocabCntr[vocab[tempStr[1]]] + 1
        else:
            vocab[tempStr[1]] = vCntr
            vocabIdx[vCntr] = tempStr[1]
            vocabCntr[vCntr] = 1
            labels.append(vocab[tempStr[1]])
            vCntr+=1

print(len(labels), ' ', featMat.shape[0])
assert len(labels)==featMat.shape[0]

print('Reading Query File')
qList=[]
with open(queryFile) as qFile:
    for line in qFile:
        qList.append(int(line))   #-1 since file was created using indexing starting from 1

qMat = featMat[tuple(qList),:]
labelArr = np.array(labels)

if args.removeQuery:
    featMat = np.delete(featMat,qList,axis=0)
    qLabelArr = labelArr[qList]
    labelArr = np.delete(labelArr,qList)
    wordPaths = np.delete(wordPaths, qList).tolist()

print('Building KD tree')
kdt = KDTree(featMat, leaf_size=30, metric='euclidean')

print('Querying KD tree')
dist, ind = kdt.query(qMat, k=featMat.shape[0], return_distance=True)

running_ap = 0.0
cntr = 1.0
for iCntr in range(len(qList)):
    qCntr = qList[iCntr]

    if args.removeQuery:
        qLabel = qLabelArr[iCntr]
    else:
        qLabel = labelArr[qCntr]
    y_true = np.zeros((dist.shape[1]))
    indices = np.where(labelArr==qLabel)

    y_true[np.where(labelArr==qLabel)] = 1
    y_true = y_true[ind[iCntr,:]]
    maxVal = np.max(dist[iCntr,:])

    if args.removeQuery:
        currAP = label_ranking_average_precision_score([y_true], [maxVal - dist[iCntr,:]])
    else:
        currAP = label_ranking_average_precision_score([y_true[1:]], [maxVal - dist[iCntr,1:]])

    running_ap+= currAP
    cntr+=1
    print('AP for Query:%d Text:%s Occ:%d is %.4f'%(qCntr,vocabIdx[qLabel],vocabCntr[qLabel],currAP))

    with open(logs_dir+'retrieval_accuracy.txt','a+') as fLog:
        fLog.write('AP for Query:%d Text:%s Occ:%d is %.4f \n'
        % (qCntr,vocabIdx[qLabel],vocabCntr[qLabel],currAP))

    if args.printFlag:
        with open(os.path.join(ret_dir,'CombAcc.txt'),'a+') as fLog:
            fLog.write('%d %d %d %s %s %.6f %.6f %.6f %.6f %.6f %d \n'
            % (iCntr+1,vocabCntr[qLabel]-1, len(vocabIdx[qLabel]), vocabIdx[qLabel],
            wordPaths[ind[iCntr,0]], currAP,0.0,0.0,0.0,0.0,0))

        with open(os.path.join(ret_dir,'OutList',str(iCntr+1)),'w') as fLog:

            for pCntr in range(np.min((args.outSize, featMat.shape[0]))):
                fLog.write('%s:%.6f \n' % (wordPaths[ind[iCntr,pCntr]], dist[iCntr,pCntr]))

mAP = running_ap/(cntr-1)
print('mAP for entire testset is %.4f'%(mAP))
with open(logs_dir+'retrieval_accuracy.txt','a+') as fLog:
        fLog.write('mAP for entire testset is %.4f \n' % (mAP))
