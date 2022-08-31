import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.init as init
from model import FFN, Model, CLSTM, PointHWnet
from torch import optim
from utils import load_csv,load_npy, Averager, CTCLabelConverter
#from Pointnet.models.pointnet2_cls_msg_ import Pointnet2
#from Pointnet.models.pointnet2_cls_ssg import Pointnet
from model import Mdl
import numpy as np
from Levenshtein import distance
import random, argparse, string, os
import json
from collections import OrderedDict
import gc
from resnet import ResNet

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def load_model_json(model,str, path, i = 0):
  data_dict = OrderedDict()
  with open(path, 'r') as f:
    data_dict = json.load(f)
  own_state = model.state_dict()
  count = 0
  for k in own_state:
    #print(count, " ",k)
    count +=1
  for k, v in data_dict.items():
    if i == 1:
      k = k[7:]
    else:
      k = str+k
    #print('Loading parameter:', k)
    if not k in own_state:
      #print('Parameter', k, 'not found in own_state!!!')
      continue
    if type(v) == list or type(v) == int:
      v = torch.tensor(v)
    own_state[k].copy_(v)
    own_state[k].requires_grad = False
  model.load_state_dict(own_state)
  print('Model loaded')


def get_data(model,cp1_val,cp2_val,type):
    output = np.zeros((1,32*95))
    with torch.no_grad():
        model.eval()
        data_hw = iter(cp1_val)
        for i, data in enumerate(cp2_val):
            image_tensors = data
            image = image_tensors.float().to(device)
            batch_size = image.size(0)
            image = image.transpose(2,1)
            preds,out = model(image, next(data_hw).float().to(device))
            output = np.append(output,out.reshape((out.shape[0],out.shape[1]*out.shape[2])),axis=0)

            print(output.shape)
    np.save('/net/voxel03/misc/extra/data/hrishikesh/points_results_'+type+'.npy',output[1:])


def train(opt):
    """ dataset preparation """

    print("inside train")

    # loading data
    #cp, l_train = load_csv('./dataset/sampled_points_train.csv','./dataset/word_text_train.csv',i=1)
    cp1 = np.load('../../../extra/data/hrishikesh/feats_original_hwnet_train.npy')
    #cp1 = np.load('/net/voxel03/misc/extra/data/hrishikesh/points_hwnet_train.npy')
    #cp2 = np.load('../../../extra/data/hrishikesh/points_new_train.npy')
    #cp2 = cp2.reshape((cp2.shape[0],1280,6))
    cp2 = np.load('/net/voxel03/misc/extra/data/hrishikesh/semantic_rep_image_train.npy')
    #cp1 = np.zeros(cp2.shape)
    #cp1 = cp1.reshape((cp1.shape[0],128,640))
    #cp = np.hstack((cp1,cp2))
    l_train = np.load('../../../extra/data/hrishikesh/text_new_train.npy')
    cp1 = torch.tensor(cp1)
    cp2 = torch.tensor(cp2)
    #cp1 = cp1.permute(0,2,1)
    #cp2 = cp2.permute(0,2,1)
    #cp = cp[:,:,:3]
    print('train data loaded',cp1.size()," ",cp2.size())


    # Create Dataloader of the above tensor with batch size
    cp1_train = DataLoader(cp1, batch_size=opt.batch_size)
    cp2_train = DataLoader(cp2, batch_size=opt.batch_size)

    #cp, l_val = load_csv('./dataset/sampled_points_val.csv','./dataset/word_text_val.csv',i=1)
    cp1 = np.load('../../../extra/data/hrishikesh/feats_original_hwnet_val.npy')
    #cp1 = np.load('/net/voxel03/misc/extra/data/hrishikesh/points_hwnet_val.npy')
    #cp2 = np.load('../../../extra/data/hrishikesh/points_new_val.npy')
    #cp2 = cp2.reshape((cp2.shape[0],1280,6)) #/1356
    cp2 = np.load('/net/voxel03/misc/extra/data/hrishikesh/semantic_rep_image_val.npy')
    #cp1 = cp1.reshape((cp1.shape[0],128,640))
    #cp = np.hstack((cp1,cp2))
    #cp1 = np.zeros(cp2.shape)
    l_val = np.load('../../../extra/data/hrishikesh/text_new_val.npy')
    cp1 = torch.tensor(cp1)
    cp2 = torch.tensor(cp2)
    #cp1 = cp1.permute(0,2,1)
    #cp2 = cp2.permute(0,2,1)
    #cp = cp[:,:,:3]
    print('val data loaded')


    # Create Dataloader of the above tensor with batch size
    cp1_val = DataLoader(cp1, batch_size=opt.batch_size)
    cp2_val = DataLoader(cp2, batch_size=opt.batch_size)



    """ model configuration """
    
    converter = CTCLabelConverter(opt.character)

    opt.num_class = len(converter.character)

    model = Mdl(opt)
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

    #load_model_json(model,"",'../../../extra/data/hrishikesh/best_model.json')
    #load_model_json(model,"Pointnet.",'./Pointnet/best_model.json')

    count = 0
    for name, param in model.named_parameters():
        print(count, ' ' ,name)
        if count  < 84: # or ( count > 88 and count < 92) :
            break
            count += 1
            continue
        #if 'Pred' in name:
            #break
            #count += 1
            #continue
        print(name)
        param.requires_grad = False
        count +=1

    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    #load_model_json(model,"",'../../../extra/data/hrishikesh/best_model_phw.json')
    #load_model_json(model,"Pointnet.",'./Pointnet/best_model.json')


    """ setup loss """
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    
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

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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


    #get_data(model,cp1_train,cp2_train,'train')
    #get_data(model,cp1_val,cp2_val,'val')

    for epoch in range(opt.num_iter):
        #break
        model.train()
        prev = final_count
        final_count = 0
        data_hw = iter(cp1_train)
        for i, data in enumerate(cp2_train):
            image_tensors = data
            labels = l_train[i*opt.batch_size:i*opt.batch_size+opt.batch_size]
            image = image_tensors.float().to(device)
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)
            #image = image.transpose(2,1)
            preds = model(image,next(data_hw).float().to(device))
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)

            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
            print(preds_str[:5], " ", labels[: 5])

            preds = preds.log_softmax(2).permute(1, 0, 2)
            cost = criterion(preds, text, preds_size, length)
            print(i," epoch : ", epoch, " val : ", min_val_loss,"in epoch : ",in_epoch," prev val : ",prev_val, " train : ", prev)
            print('cost : ', cost )

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()

            #loss_avg.add(cost)

            count = 0
            dist = 0
            for i in range(len(labels)):
                #dist += distance(preds_str[i],labels[i])
                if preds_str[i] == labels[i]:
                    count += 1

            final_count += count
            #avg_dist = dist/opt.batch_size
            print("Correct Words Predicted : ", count,'/',opt.batch_size)
            #print('Avg Levenshtein Distance : ',avg_dist)

            # save model per 1e+5 iter.
            #if (epoch + 1) % 50 == 0:
        torch.save(
            model.state_dict(), f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}/iter_new_lstm.pth')

        #if not opt.val:
        #    continue
        scheduler.step()
        with torch.no_grad():
            model.eval()
            count_val  = 0
            data_hw = iter(cp1_val)
            for i, data in enumerate(cp2_val):
                # val part
                #if len(data) != opt.batch_size:
                    #continue
                image_tensors = data
                labels = l_val[i*opt.batch_size:i*opt.batch_size+opt.batch_size]
                image = image_tensors.float().to(device)
                text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
                batch_size = image.size(0)
                #image = image.transpose(2,1)
                preds = model(image, next(data_hw).float().to(device))
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)

                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)
                print(preds_str[:5], " ", labels[: 5])

                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

                print(i ,' cost : ', cost)


            #loss_avg_val.add(cost)
                for i in range(len(labels)):
                    if(preds_str[i]==labels[i]):
                        count_val+=1

        prev_val = count_val
        print(prev_val)
        #break
        if min_val_loss < count_val:
            in_epoch = epoch
            min_val_loss = count_val
            torch.save(
                model.state_dict(), f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}/best_model_new_lstm_0.3.pth')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
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
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=50, help='maximum-label-length')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz\'/\\.,?"(){}[]-_=+@!$%:#&*;', help='character label')

    """ Model Architecture """
    parser.add_argument('--input_size', type=int, default=96, help='the size of the LSTM hidden state')
    parser.add_argument('--hidden_size', type=int, default=128, help='the size of the LSTM hidden state')
    parser.add_argument('--out', nargs="+", default=[128,256], help='number of iterations to train for')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if not opt.exp_name:
        opt.exp_name = 'model_hwnet_original_text_image'
        # print(opt.exp_name)


    #opt.saved_model = "../../../extra/data/hrishikesh/saved_models/model_hwnet_original_text/best_model_4921.pth"

    os.makedirs(f'../../../extra/data/hrishikesh/saved_models/{opt.exp_name}', exist_ok=True)

    opt.character = string.printable[:-6]
    opt.character = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    print(len(opt.character))

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    opt.num_gpu = torch.cuda.device_count()

    train(opt)
