import torch
from torch.utils.data import DataLoader
from utils import load_csv
from model import FFN
import numpy as np
import argparse
from torch_points3d.applications.pretrained_api import PretainedRegistry


def train(opt):
    """ dataset preparation """

    # loading data
    cp, l_train = load_csv('./dataset/points_2.csv','./dataset/word_text_train.csv')

    for cloud in cp:
        for point in cloud:
            point = np.append(point, np.zeros((1)))
    cp = torch.tensor(cp)
    print('train data loaded',cp.size())


    # Create Dataloader of the above tensor with batch size
    cp_train = DataLoader(cp, batch_size=opt.batch_size)
    
    """ model configuration """

    model = PretainedRegistry.from_pretrained("pointnet2_largemsg-s3dis-1")

    model = model.to(device)
    model.eval()

    print("Model:")
    print(model)


    for epoch in range(opt.num_iter):

        for i, data in enumerate(cp_train):
            # train part
            image_tensors = data
            image = image_tensors.float().to(device)
            
            preds = model(image)
            print(preds.shape)
            print(preds)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=1, help='number of iterations to train for')
    """ Data processing """
    
    opt = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    opt.num_gpu = torch.cuda.device_count()

    train(opt)
