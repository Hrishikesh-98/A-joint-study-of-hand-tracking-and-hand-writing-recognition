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
from utils import load_csv,load_npy, Averager, CTCLabelConverter, getVocab
#from Pointnet.models.pointnet2_cls_msg_ import Pointnet2
#from Pointnet.models.pointnet2_cls_ssg import Pointnet
from model import Mdl7 as Mdl
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

    transformer = transforms.Compose([
    synthTrans.Normalize(),
    synthTrans.ToTensor() ])

    tSize = 48
    
    train_vocab = getVocab(ann_file=opt.trainAnnFile)
    val_vocab = getVocab(ann_file=opt.valAnnFile)

    #cp1 = hwROIDat.HWRoiDataset(ann_file=opt.trainAnnFile,
    #                                    img_folder=opt.img_folder,
    #                                    randFlag=False,
    #                                    valFlag = False,
    #                                    transform=transformer,
    #                                    testFontSize=tSize)


    # loading data
    #cp, l_train = load_csv('./dataset/sampled_points_train.csv','./dataset/word_text_train.csv',i=1)
    #cp2 = np.load('/net/voxel03/misc/extra/data/hrishikesh/points_new_train.npy')
    cp1 = np.load('../../../extra/data/hrishikesh/feats_original_hwnet_train.npy')
    temp = np.load('../../../extra/data/hrishikesh/feats_original_hwnet_val.npy')
    cp1 = np.append(cp1,temp,axis=0)
    #cp2 = cp2.reshape((cp2.shape[0],1280,6))
    l_vecs = np.load('../../../extra/data/hrishikesh/group_rep_new_train.npy')
    temp = np.load('../../../extra/data/hrishikesh/group_rep_new_val.npy')
    l_vecs = np.append(l_vecs,temp,axis=0)
    l_vecs = l_vecs.reshape((l_vecs.shape[0],128,640))
    l_vecs = torch.tensor(l_vecs)
    print("1")
    l_points = np.load('../../../extra/data/hrishikesh/sample_points_train.npy')
    temp = np.load('../../../extra/data/hrishikesh/sample_points_val.npy')
    l_points = np.append(l_points,temp,axis=0)
    l_points = l_points.reshape((l_points.shape[0],128,3))
    l_points = torch.tensor(l_points)
    print("2")
    cp2 = torch.cat((l_points,l_vecs),dim=2)
    print("3")
    l_train = np.load('/net/voxel03/misc/extra/data/hrishikesh/trainval1_new_labels_lower.npy')
    #l_train = np.load('../../../../../../misc/extra/data/hrishikesh/text_new_train.npy')
    cp1 = torch.tensor(cp1)
    l_vecs = None
    l_points = None
    #temp = None
    #cp2 = torch.tensor(cp2)
    #cp1 = cp1.permute(0,2,1)
    #cp2 = cp2.permute(0,2,1)
    #cp = cp[:,:,:3]
    l_train = torch.tensor(l_train)
    print('train data loaded'," ",cp2.size(), type(cp1))


    # Create Dataloader of the above tensor with batch size
    cp1_train = DataLoader(cp1, batch_size=opt.batch_size)
    cp2_train = DataLoader(cp2, batch_size=opt.batch_size)

    #cp1 = hwROIDat.HWRoiDataset(ann_file=opt.valAnnFile,
    #                                    img_folder=opt.img_folder,
    #                                    randFlag=False,
    #                                    valFlag = True,
    #                                    transform=transformer,
    #                                    testFontSize=tSize)
    #cp, l_val = load_csv('./dataset/sampled_points_val.csv','./dataset/word_text_val.csv',i=1)
    #cp2 = np.load('/net/voxel03/misc/extra/data/hrishikesh/points_new_val.npy')
    cp1 = np.load('../../../extra/data/hrishikesh/feats_original_hwnet_val2.npy')
    #cp2 = cp2.reshape((cp2.shape[0],1280,6)) #/1356
    l_vecs = np.load('../../../extra/data/hrishikesh/group_rep_new_val2.npy')
    l_vecs = l_vecs.reshape((l_vecs.shape[0],128,640))
    l_vecs = torch.tensor(l_vecs)
    l_points = np.load('../../../extra/data/hrishikesh/sample_points_val2.npy')
    l_points = l_points.reshape((l_points.shape[0],128,3))
    l_points = torch.tensor(l_points)
    cp2 = torch.cat((l_points,l_vecs),dim=2)
    #cp1 = cp1.reshape((cp1.shape[0],128,640))
    #cp = np.hstack((cp1,cp2))
    #cp1 = np.zeros(cp2.shape)
    l_val = np.load('/net/voxel03/misc/extra/data/hrishikesh/val2_new_labels_lower.npy')
    #l_val = np.load('../../../../../../misc/extra/data/hrishikesh/text_new_val.npy')
    l_vecs = None
    l_points = None
    cp1 = torch.tensor(cp1)
    #cp2 = torch.tensor(cp2)
    #cp1 = cp1.permute(0,2,1)
    #cp2 = cp2.permute(0,2,1)
    #cp = cp[:,:,:3]
    l_val = torch.tensor(l_val)
    print('val data loaded')


    # Create Dataloader of the above tensor with batch size
    cp1_val = DataLoader(cp1, batch_size=opt.batch_size)
    cp2_val = DataLoader(cp2, batch_size=opt.batch_size)

    model = Mdl(opt)
    print(model)

    #load_model_json(model,"PointnetFeatureExtraction.",'./Pointnet/best_model.json')

    feats_train = []

    for i,data in enumerate(cp1_train):
        feats_train.append(data)
        #if i == 2028:
        #    print(data['label'])

    feats_val = []

    for data in cp1_val:
        feats_val.append(data)


    """ model configuration """
    
    converter = CTCLabelConverter(opt.character)

    opt.num_class = len(converter.character)

    #model = Mdl(opt)
    #model = nn.DataParallel(model, device_ids = [0,1])
    #model = Pointnet(opt.num_class)
    print('model input parameters', opt.input_size, opt.hidden_size,opt.num_class, opt.batch_max_length)


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
    model.train()

    #load_model_json(model,"module.",'/net/voxel03/misc/extra/data/hrishikesh//best_ctc.json',1)
    #load_model_json(model,"FeatureExtraction.",'./best_model.json')

    count = 0
    '''for name, param in model.named_parameters():
        if count  < 92: # or ( count > 88 and count < 92) :
            break
            count += 1
            continue
        if 'Pred' in name:
            #break
            count += 1
            continue
        print(name)
        param.requires_grad = False
        count +=1'''

    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    """ setup loss """
    #criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    criterion = nn.CrossEntropyLoss()
    loss_avg = Averager()
    loss_avg_val = Averager()
    min_val_loss = 0
    final_count  = 0
    prev_t = 0
    prev_val =0

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    iteration = start_iter
    in_epoch = 0

    #torch.save(
    #    model.state_dict(), f'net/voxel03/misc/extra/data/hrishikesh/saved_models/{opt.exp_name}/iter_new_both.pth')

    #torch.save(
        #model.state_dict(), f'net/voxel03/misc/extra/data/hrishikesh//saved_models/{opt.exp_name}/best_model_both.pth')

    #get_data(model,feats_train,cp2_train,'train')
    #get_data(model,feats_val,cp2_val,'val')


    for epoch in range(opt.num_iter):
        #break
        model.train()
        final_count = 0
        for i, data in enumerate(cp2_train):
            #break
            image_tensors = data
            data_hw = feats_train[i]
            labels = l_train[i*opt.batch_size:i*opt.batch_size+opt.batch_size]
            #labels = data_hw['label']
            image = image_tensors.float().to(device)
            #text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)
            image = image.transpose(2,1)
            preds,_ = model(image,data_hw.float().to(device))
            #preds_size = torch.IntTensor([preds.size(1)] * batch_size)

            #_, preds_index = preds.max(2)
            #preds_str = converter.decode(preds_index, preds_size)
            #print(preds[:5], " ", labels[: 5])

            _,preds_str = preds.log_softmax(1).max(1)
            #labels = labels.squeeze()
            for p in range(5):
                print(train_vocab[preds_str[p].item()], " ", train_vocab[labels[p].item()])
            cost = criterion(preds, labels.long().cuda())
            print(i," epoch : ", epoch, " val : ", min_val_loss,"in epoch : ",in_epoch," prev val : ",prev_val, " train : ", prev_t)
            print('cost : ', cost )

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)
            count = 0
            for j in range(len(labels)): #.squeeze())):
                if(preds_str[j]==labels[j].cuda()):
                    count+=1

            final_count += count
            #avg_dist = dist/opt.batch_size
            #print("Correct Words Predicted : ", count,'/',opt.batch_size)
            #print('Avg Levenshtein Distance : ',avg_dist)

            # save model per 1e+5 iter.
            #if (epoch + 1) % 50 == 0:
        torch.save(
            model.state_dict(), f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}/iter_new_both.pth')

        prev_t = final_count
        #if not opt.val:
        #    continue
        #scheduler.step()
        with torch.no_grad():
            model.eval()
            count_val  = 0
            for i, data in enumerate(cp2_val):
                # val part
                #if len(data) != opt.batch_size:
                    #continue
                image_tensors = data
                data_hw = feats_val[i]
                labels = l_val[i*opt.batch_size:i*opt.batch_size+opt.batch_size]
                #labels = data_hw['label']
                image = image_tensors.float().to(device)
                #text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
                batch_size = image.size(0)
                image = image.transpose(2,1)
                preds,_ = model(image, data_hw.float().to(device))
                #preds_size = torch.IntTensor([preds.size(1)] * batch_size)

                _, preds_str = preds.log_softmax(1).max(1)
                #preds_str = converter.decode(preds_index, preds_size)
                #labels = labels.squeeze()
                #for p in range(len(preds_str[:5])):
                #    print(val_vocab[preds_str[p].item()], " ", val_vocab[labels[p].item()])

                #preds = preds.log_softmax(2).permute(1, 0, 2)
                #cost = criterion(preds, labels.long().cuda())

                #print(i ,' cost : ', cost)


                #loss_avg_val.add(cost)
                for j in range(len(labels)):
                    if(preds_str[j].item() == labels[j].item()):
                        count_val+=1
                print(i, ' ', count_val)

        prev_val = count_val
        if min_val_loss < prev_val:
            in_epoch = epoch
            min_val_loss = prev_val
            torch.save(
                model.state_dict(), f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}/best_model_both_'+str(min_val_loss)+'.pth')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
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

    """ HwNet """
    parser.add_argument('--img_folder', default='../../../../../../misc/extra/data/hrishikesh/words_dataset/', help='image root folder')
    parser.add_argument('--trainAnnFile', default='hwnet/ann/train_new_ann.txt')
    parser.add_argument('--valAnnFile', default='hwnet/ann/val_new_ann.txt')
    parser.add_argument('--exp_dir', default='output/')
    parser.add_argument('--testAug', action='store_true', default=False, help='perform test side augmentation')
    parser.add_argument('--pretrained_file', default='pretrained/iam-model.t7', help='pre trained file path')
    parser.add_argument('--exp_id', default='iam-test-0', help='experiment ID')

    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=50, help='maximum-label-length')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz\'/\\.,?"(){}[]-_=+@!$%:#&*;', help='character label')

    """ Model Architecture """
    parser.add_argument('--input_size', type=int, default=512, help='the size of the LSTM hidden state')
    parser.add_argument('--hidden_size', type=int, default=128, help='the size of the LSTM hidden state')
    parser.add_argument('--out', nargs="+", default=[128,256], help='number of iterations to train for')

    opt = parser.parse_args()
    print(opt.img_folder)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(device)

    if not opt.exp_name:
        opt.exp_name = 'model_hwpoint_original_label_lower_trainval1'
        # print(opt.exp_name)


    #opt.saved_model = "../../../extra/data/hrishikesh/saved_models/model_hwpoint_original_label_lower_128_x2/best_model_both_5247.pth"

    os.makedirs(f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}', exist_ok=True)

    opt.character = string.printable[:-6]

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)

    opt.num_gpu = torch.cuda.device_count()

    train(opt)
