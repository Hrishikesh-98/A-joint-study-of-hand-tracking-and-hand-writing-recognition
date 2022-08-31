import numpy as np
from utils import load_csv
import csv

x, train = load_csv('./dataset/feat_vec_train.csv','./dataset/word_text_train.csv',False,i=1)
y, test = load_csv('./dataset/sampled_points_train.csv','./dataset/word_text_train.csv',True,i=1)

print(x.shape)
print(y.shape)

y = y[:,:,:2]
y = y.reshape((y.shape[0],256))

print(x.shape)
print(y.shape)

new = np.hstack((x,y))

print(new.shape)

np.save('./dataset/merged_train.npy',new)
