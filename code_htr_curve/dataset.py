import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import torch
import os

def get_data(annfile,root):
    f = open(annfile,"r")
    lines = f.readlines()
    images = []
    for i,line in enumerate(lines):
        print(i)
        val = line.split()[0]
        print(root+val)
        ext = '.png'
        if 'train2' in annfile:
           ext = ''
        image = cv2.imread(root+val+ext)
        #print(image.shape)
        if "mask" in root:
            #print("val")
            image = cv2.resize(image, (256,256))
            word_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #ret, img = cv.threshold(img, 150, 255, 1)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(256,256))
            input_image = Image.fromarray(image)
            preprocess = transforms.Compose([
                    #transforms.Resize(256),
                    #transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            input_tensor = preprocess(input_image)
            word_image = input_tensor.numpy()
        #print(word_image.shape)
        images.append(word_image)
    f.close()
    return images

def pad_points(points, seq_len):
    for i in range(len(points)):
        #print(points[i].shape)
        points_size = points[i].shape[0]
        pad = torch.tensor(np.array([[256,256]]*(seq_len-points_size)))
        #print(pad.shape)
        points[i] = torch.cat((points[i],pad),dim=0)
        #print(points[i].shape)

    return points

def get_points(annfile,root):
    f = open(annfile,"r")
    lines = f.readlines()
    points = []
    maxi = 0
    for line in lines:
        val = line.split()[0]
        print(i, ' ',root+val)
        image = cv2.imread(root+val) #+".png")
        image = cv2.resize(image, (256, 256))
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 150, 255, 1)
        invert = cv2.bitwise_not(thresh)
        contours, hierarchy = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(invert, contours, -1, (255,255,255), 1)
        all_points = np.zeros((1,2))

        for h in range(invert.shape[0]):
            for w in range(invert.shape[1]):
                if invert[h][w] == 255:
                    temp = np.zeros((1,2))
                    temp[0][0] = w
                    temp[0][1] = h
                    all_points = np.append(all_points,temp,axis=0)

        temp = np.array([[256,256]])
        all_points = np.append(all_points,temp,axis=0)
        maxi = max(maxi, all_points.shape[0])
        points.append(torch.tensor(all_points))
    print(maxi)
    if 'test' in annfile or 'val' in annfile:
        points = pad_points(points,maxi)
    return points

def read_video(vid_number,root):
    #cap= cv2.VideoCapture(root+str(vid_number)+'.mp4')
    #print(root+str(vid_number))
    files = os.listdir(root+'/'+str(vid_number)+'/')
    #print(files)
    images = []
    for file in files:
        image = cv2.imread(root+'/'+str(vid_number)+'/'+file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(256,256))
        input_image = Image.fromarray(image)
        preprocess = transforms.Compose([
                    #transforms.Resize(256),
                    #transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        input_tensor = preprocess(input_image)
        #print(input_tensor.shape)
        images.append(input_tensor)

    return images
