import numpy as np
import os
from svgpathtools import svg2paths
import csv

def get_size(svg):
  pth = '../npzs_test/clean_line_drawings__pretrain_clean_line_drawings/seq_data/svgs/'+svg

  paths, attributes = svg2paths(pth)
  return len(attributes)


def get_controls(svg):
  pth = '../npzs_test/clean_line_drawings__pretrain_clean_line_drawings/seq_data/svgs/'+svg

  paths, attributes = svg2paths(pth)
  stroke_num = len(attributes)
  controls = np.zeros((stroke_num,7))
  if len(attributes) > 0:
      start_x = float(attributes[0]['d'].split()[1][:-1])
      start_y = float(attributes[0]['d'].split()[2][:-1])
  for k, v in enumerate(attributes):
      #print(v)
      nums = v['d'].split()
      controls[k][0] = float(float(nums[1][:-1])-start_x)
      controls[k][1] = float(float(nums[2][:-1])-start_y)
      controls[k][2] = float(float(nums[4][:-1])-start_x)
      controls[k][3] = float(float(nums[5][:-1])-start_y)
      controls[k][4] = float(float(nums[6][:-1])-start_x)
      controls[k][5] = float(float(nums[7][:-1])-start_y)
      controls[k][6] = float(v['stroke-width'])

  controls = controls[np.argsort(controls[:,0])]
  controls = np.append(controls,np.zeros((128-stroke_num,7)),axis = 0)
  return controls

path = '../npzs_test/clean_line_drawings__pretrain_clean_line_drawings/seq_data/svgs/'

x = os.listdir(path)

embeddings_dict = {}
with open("../glove.6B.50d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:],"float32")
        embeddings_dict[word] = vector


data = open('../dataset_test.txt','r')
dataset =data.readlines()
data.close()

control_points = []
word_labels = []

for line in dataset:
  values = line.split()
  name = values[0].split('/')[-1]
  label = values[1]
  if name + '.svg' in x and get_size(name+'.svg') < 128:
    control_points.append(get_controls(name+'.svg'))
    word_labels.append(label )


with open('./control_points_test_norm.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(np.array(control_points))

with open('./word_text_test_norm.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(word_labels)
