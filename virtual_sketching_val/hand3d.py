import numpy as np

embeddings_dict = {}
with open("../glove.6B.50d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

f = open('../dataset.txt','r')
f1 = open('../labels.txt','a')
labels = []
lines = f.readlines()

import os

for line in lines[938:]:
  values = line.split()
  if values[1] in embeddings_dict.keys():
    print(filename)
    filename = '../words_dataset/'+values[0] + '.png'
    cmd = 'python3 test_vectorization.py --input ' + filename
    os.system(cmd)
    print('     vectorization done')
    npz_path = '../npzs/clean_line_drawings__pretrain_clean_line_drawings/seq_data/' + values[0].split('/')[-1] + '.npz'
    cmd = 'python3 ./tools/svg_conversion.py --file ' + npz_path
    os.system(cmd)
    print('     svg conversion done')
    a = values[1] + ' 1\n'
    f1.write(labels)

f.close()
f1.close()
