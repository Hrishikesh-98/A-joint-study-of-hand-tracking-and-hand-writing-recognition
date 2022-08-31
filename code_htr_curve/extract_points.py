import numpy as np
import os
from svgpathtools import svg2paths
import csv
import math


max = 0
min = np.inf

def getValue(x1,x2,t):
    return x1+(x2-x1)*t


def get_size(svg):
  #pth = '../npzs_val/clean_line_drawings__pretrain_clean_line_drawings/seq_data/svgs/'+svg
  pth = '../../../extra/data/hrishikesh/npzs_stroke_256_test/clean_line_drawings__pretrain_clean_line_drawings/seq_data/svgs/'+svg

  paths, attributes = svg2paths(pth)
  return len(attributes)


def get_controls(svg):
  #pth = '../npzs_val/clean_line_drawings__pretrain_clean_line_drawings/seq_data/svgs/'+svg
  pth = '../../../extra/data/hrishikesh/npzs_stroke_256_test/clean_line_drawings__pretrain_clean_line_drawings/seq_data/svgs/'+svg


  paths, attributes = svg2paths(pth)
  stroke_num = len(attributes)
  points = np.zeros((stroke_num*10,6))
  global max
  global min
  for k, v in enumerate(attributes):
      #print(v)
      nums = v['d'].split()
      x0 = (float(nums[1][:-1]))
      if x0 > max:
        max = x0
      y0 = (float(nums[2][:-1])) #*-1
      if y0 < min:
        min = y0
      x1 = (float(nums[4][:-1]))
      if x1 > max:
        max = x1
      y1 = (float(nums[5][:-1])) #*-1
      if y1 < min:
        min = y1
      x2 = (float(nums[6][:-1]))
      if x2 > max:
        max = x2
      y2 = (float(nums[7][:-1])) #*-1
      if y2 < min:
        min = y2
      if x0 > x2:
        temp1 = x2
        temp2 = y2
        x2 = x0
        y2 = y0
        x0 = temp1
        y0 = temp2
      tmp = 1/10
      for i in range(10):
        t = i * tmp
        x = (1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2
        y = (1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2
        d1x = 2 * (x1 - x0)
        d1y = 2 * (y1 - y0)
        d2x = 2 * (x2 - x1)
        d2y = 2 * (y2 - y1)
        dx = (1 - t) * d1x + t * d2x
        dy = (1 - t) * d1y + t * d2y
        #dx = (int)((2*t-2) * x0 + t *(2*x2-4*x1)  + 2 * x1)
        #dy = (int)((2*t-2) * y0 + t *(2*y2-4*y1)  + 2 * y1) 
        q = math.sqrt(dx * dx + dy * dy);
        nx = -dy / q;
        ny = dx / q;
        nz = 0.0
        points[k*10+i][0] = x
        points[k*10+i][1] = y
        points[k*10+i][2] = 0
        points[k*10+i][3] = nx
        points[k*10+i][4] = ny
        points[k*10+i][5] = nz

  #points = points[np.argsort(points[:,0])]
  zeros = np.zeros((5120-stroke_num*10,6))
  print(zeros.shape)
  points = np.append(points,zeros,axis=0 )
  print(points.shape)
  print(points)
  return points



#path = '../npzs_val/clean_line_drawings__pretrain_clean_line_drawings/seq_data/svgs/'
path = '../../../extra/data/hrishikesh/npzs_stroke_256_test/clean_line_drawings__pretrain_clean_line_drawings/seq_data/svgs/'


x = os.listdir(path)

print(len(x))

f = open('./dataset/hw.txt','r')
sizes= f.readlines()
max_h = float(sizes[0].split()[0])
max_w = float(sizes[1].split()[0])
mean_h = float(sizes[2].split()[0])
mean_w = float(sizes[3].split()[0])
data = open('../../../extra/data/hrishikesh/hwnet-master/ann/test_stroke_ann.txt','r')
dataset =data.readlines()
data.close()

ann_txt = open('../hwnet/ann/try_stroke_ann.txt','w')
bnn_txt = open('../tryy_ann2.txt','w')

curve_points = []
labels = []

count = 0
m = 0
for line in dataset:
  values = line.split()
  name = values[0].split('/')[-1].split(".")[0]
  label = values[1]
  print(name)
  parts = name.split('-')
  s = get_size(name + '.svg')
  m = m if m > s else s
  if name + '.svg' in x and get_size(name+'.svg') < 512:
    curve_points.append(get_controls(name+'.svg'))
    labels.append(label)
    print(label)
    vals = line.split()
    ann_txt.write(vals[0]+'.png '+vals[1]+ ' '+vals[2]+' 1\n')
    bnn_txt.write(vals[0]+' '+vals[1]+ ' '+vals[2]+' 1\n')
  print(m)

print(max)
print(min)
cp = np.array(curve_points)
c = cp.reshape((cp.shape[0],5120*6))
print(c.shape)
np.save('/net/voxel03/misc/extra/data/hrishikesh/points_stroke_256_test.npy',c)
#x = np.load('/net/voxel03/misc/extra/data/hrishikesh/semantic_in_train_1.npy')
#y = np.load('/net/voxel03/misc/extra/data/hrishikesh/semantic_in_train_2.npy')


#np.save('/net/voxel03/misc/extra/data/hrishikesh/semantic_in_train1.npy',np.append(x,y,axis=0))

#x = np.load('/net/voxel03/misc/extra/data/hrishikesh/feats_train.npy')
#y = np.load('/net/voxel03/misc/extra/data/hrishikesh/feats_train2.npy')

#np.save('/net/voxel03/misc/extra/data/hrishikesh/feats_trainx2.npy',np.append(x,y,axis=0))

#x = np.load('/net/voxel03/misc/extra/data/hrishikesh/text_normals_train.npy')
#y = np.load('/net/voxel03/misc/extra/data/hrishikesh/text_normals_train2.npy')

#np.save('/net/voxel03/misc/extra/data/hrishikesh/text_normals_trainx2.npy',np.append(x,y,axis=0))


#np.save('/net/voxel03/misc/extra/data/hrishikesh/text_new2_val.npy',np.array(labels))

#with open('./dataset/a01-000x.csv', 'w') as f:
#    write = csv.writer(f)
#    write.writerows(np.array(curve_points))

ann_txt.close()
bnn_txt.close()
