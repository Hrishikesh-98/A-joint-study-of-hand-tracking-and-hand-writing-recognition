import numpy as np
import os
from svgpathtools import svg2paths
import csv
import math


def getValue(x1,x2,t):
    return x1+(x2-x1)*t


def get_size(pth):

  paths, attributes = svg2paths(pth)
  return len(attributes)


def get_controls(pth,j):

  paths, attributes = svg2paths(pth)
  stroke_num = len(attributes)
  points = np.zeros((stroke_num*10,6))
  for k, v in enumerate(attributes):
      #print(v)
      nums = v['d'].split()
      x0 = (float(nums[1][:-1]))
      y0 = (float(nums[2][:-1])) *-1
      x1 = (float(nums[4][:-1]))
      y1 = (float(nums[5][:-1])) *-1
      x2 = (float(nums[6][:-1]))
      y2 = (float(nums[7][:-1])) *-1
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
        if j == 1:
            x = math.ceil(x)
            y = math.ceil(y)
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
        points[k*10+i][2] = 0.0
        points[k*10+i][3] = nx
        points[k*10+i][4] = ny
        points[k*10+i][5] = nz

  points = points[np.argsort(points[:,0])]
  l = len(attributes)
  pad = 128
  while l > pad:
    pad +=128
  zeros = np.zeros((pad*10-stroke_num*10,6))
  print(zeros.shape)
  points = np.append(points,zeros,axis=0 )
  print(points.shape)
  print(points)
  return points


def get_points(folder,file,i,j=0):
    path = '../../'+folder+'/clean_line_drawings__pretrain_clean_line_drawings/seq_data/svgs/'

    x = os.listdir(path)
    
    data = open('../../'+file+'.txt','r')
    dataset =data.readlines()
    data.close()

    values = dataset[i].split()
    name = values[0].split('/')[-1]


    return get_controls(path+name+'.svg',j)

