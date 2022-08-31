import cv2 as cv
import numpy as np
import torch 

def rotate_180(array, M, N):
    out = array.copy()
    for i in range(M):
        for j in range(N):
            out[i, N-1-j] = array[M-1-i, j]
            
    return out

y = np.load("labeled_mask_train.npy")[10:]
y = y.reshape((y.shape[0],256,256))
x = np.load("image_labeled_test.npy")
x = x.reshape((x.shape[0],256,256))
colorMap = {0 : [0,0,0], 1 : [255,255,255], 2: [255,0,0] , 3 : [0,255,0] , 4 : [0,0,255] , 5 : [128, 64, 0] , 6 : [ 0,82, 64], 7 : [17,0,128] ,
            8 : [91,19,61], 9 : [192,76,47], 10: [128,192,64] , 11 : [128,0,192] , 12 : [64,192,255] , 13 : [64, 128, 10] , 14 : [ 128,128, 128], 15 : [64,64,64] ,
            16 : [192,192,192], 17 : [100,0,100], 18: [100,100,0] , 19 : [0,100,100] , 20 : [32,192,64] , 21 : [64, 255, 192] , 22 : [ 255,10, 90], 23 : [48,31,45],
            24 : [9, 27, 81], 25 : [162, 69, 34]}
for k,m in enumerate(x[:15]):
    #m = np.rot90(m,1)
    #m = rotate_180(m,m.shape[1],m.shape[0]).transpose()
    img = np.zeros((256,256,3))
    orig = np.zeros((256,256,3))
    print(m.max(axis=0))
    for i in range(256):
        for j in range(256):
            img[i][j][0] = colorMap[int(m[i][j])][0]
            img[i][j][1] = colorMap[int(m[i][j])][1]
            img[i][j][2] = colorMap[int(m[i][j])][2]
    for i in range(256):
        for j in range(256):
            orig[i][j][0] = colorMap[int(y[k][i][j])][0]
            orig[i][j][1] = colorMap[int(y[k][i][j])][1]
            orig[i][j][2] = colorMap[int(y[k][i][j])][2]
    cv.imshow("image",img)
    #cv.imwrite("./model_mask.png",img)
    #cv.imshow("ima",im)
    cv.imshow("original",orig)
    #cv.imwrite("./orig_mask.png",orig)
    cv.waitKey()
    cv.destroyAllWindows()