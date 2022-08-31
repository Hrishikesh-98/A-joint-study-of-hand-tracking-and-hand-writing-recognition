import numpy as np
from utils import load_csv
import random

cp_1, l_1 = load_csv('./dataset/sampled_points_train.csv','./dataset/word_text_train.csv',i=1)
train = np.load('./dataset/points_with_normals_and_width_train.npy')
print('train loaded')

cp_2, l_2 = load_csv('./dataset/sampled_points_val.csv','./dataset/word_text_val.csv',i=1)
val = np.load('./dataset/points_with_normals_and_width_val.npy')
print('val loaded')

cp_3, l_3 = load_csv('./dataset/sampled_points_test.csv','./dataset/word_text_test.csv',i=1)
test = np.load('./dataset/points_with_normals_and_width_test.npy')
print('test loaded')


data = np.append(train, val,axis=0)
data_points = np.append(data, test, axis=0)

data = np.append(l_1, l_2,axis=0)
data_labels = np.append(data, l_3, axis=0)


joined_lists = list(zip(data_points,data_labels))
random.shuffle(joined_lists) # Shuffle "joined_lists" in place
new_x, new_y = zip(*joined_lists) # Undo joining


np.save("./dataset/data_train.npy",new_x[:40884])
np.save("./dataset/text_train.npy",new_y[:40884])

np.save("./dataset/data_val.npy",new_x[40884:46288])
np.save("./dataset/text_val.npy",new_y[40884:46288])

np.save("./dataset/data_test.npy",new_x[46288:])
np.save("./dataset/text_test.npy",new_y[46288:])


x = np.load('./dataset/text_val.npy')

print(x.shape)

print(x[:3])
