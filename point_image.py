import cv2 as cv
import numpy as np
import os

def color_mask(image):
    sums = image.sum(axis=0)
    print(sums)
    label = 1
    coloring = 0
    for i in range(256):
        print(i , ' sum is ', sums[i], ' coloring value is ', coloring, ' label is ' , label) 
        if sums[i] > 0 and coloring == 0:
            coloring = 1
        if sums[i] == 0 and coloring == 1:
            coloring = 0
            label += 1
        print("set to " , ' sum is ', sums[i], ' coloring value is ', coloring, ' label is ' , label)
        for j in range(256):
            if coloring and image[i][j] > 0:
                print("setting ",label," for all columns ",j)
                image[i][j] = label

    return image
    
def get_points_image(file):
    path = './stroke_black/'
    im = cv.imread(path+file)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 150, 255, 1)
    thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
    all_points = np.zeros((1,3))
    invert = cv.bitwise_not(thresh)
    return thresh
    
if __name__ == '__main__':
    f = open("../val_stroke_ann.txt",'r')
    lines = f.readlines()
    for i,line in enumerate(lines):
        file_name = line.split()[0].strip()
        folder = file_name.split('.')[0].split('/')[0]
        print(file_name)
        img = get_points_image(file_name)
        print(i)
        cv.imwrite("./stroke_256/"+file_name,img)