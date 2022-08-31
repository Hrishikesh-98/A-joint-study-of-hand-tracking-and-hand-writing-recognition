import cv2
import numpy as np
from plot_points import get_color_value
from point_image import get_points_image

def euclidean(x1,y1,x2,y2):
    return np.sqrt( ((x2 - x1)*(x2-x1))  + ((y2 - y1)*(y2-y1)) )

def get_nearest_label(points, colorMap):
    dist = np.inf
    label = 0
    for i in range(26):
        x1 = points[0]
        y1 = points[1]
        z1 = points[2]
        x2 = colorMap[i][0]
        y2 = colorMap[i][1]
        z2 = colorMap[i][2]
        #print(x1, ' ', y1 , ' ', z1 , ' ', x2 , ' ', y2 , ' ', z2)
        euclid_dist = np.sqrt( ((x2 - x1)*(x2-x1))  + ((y2 - y1)*(y2-y1)) + ((z2 - z1)*(z2-z1)))
        #print(dist, ' ', euclid_dist)
        if dist > euclid_dist:
            dist = euclid_dist
            label = i
    return int(label)

def color_black_image(image,labeled_point_cloud, row, col):
    colorMap = {0 : [0,0,0], 1 : [255,255,255], 2: [255,0,0] , 3 : [0,255,0] , 4 : [0,0,255] , 5 : [128, 64, 0] , 6 : [ 0,82, 64], 7 : [17,0,128] ,
                8 : [91,19,61], 9 : [192,76,47], 10: [128,192,64] , 11 : [128,0,192] , 12 : [64,192,255] , 13 : [64, 128, 10] , 14 : [ 128,128, 128], 15 : [64,64,64] ,
                16 : [192,192,192], 17 : [100,0,100], 18: [100,100,0] , 19 : [0,100,100] , 20 : [32,192,64] , 21 : [64, 255, 192] , 22 : [ 255,10, 90], 23 : [48,31,45],
                24 : [9, 27, 81], 25 : [162, 69, 34]}


    '''reverseColorMap = {}

    colors = []
    for i in range((int(labeled_point_cloud[-1][3])+1)):
        colors.append([np.inf,0])
    print((int(labeled_point_cloud[-1][3])+1))
    print(colors[0][0])
    for point in labeled_point_cloud:
        colors[int(point[3])][0] = min(int(point[0]),colors[int(point[3])][0])
        colors[int(point[3])][1] = max(int(point[0]),colors[int(point[3])][1])
        #print(point[0], ' ', point[1], ' ', int(point[3]), ' ', colors)
        #print("updated colors[",int(point[3]),"][0] with ",min(int(point[1]),colors[int(point[3])][0]),
        #        " and colors[",int(point[3]),"][1] with ",min(int(point[1]),colors[int(point[3])][1])) 


    for i in range(len(colors)):
        print(colors[i][0], ' ', colors[i][1]+1)
        for j in range(colors[i][0],colors[i][1]+1):
            for k in range(row):
                if not (image[k][j][0] == 0 and image[k][j][1] == 0 and image[k][j][2] == 0):
                    #print(i+1 , ' ', colorMap[i+1])
                    image[k][j][0] = colorMap[i+1][0]
                    image[k][j][1] = colorMap[i+1][1]
                    image[k][j][2] = colorMap[i+1][2]


    for i in range(25):
        reverseColorMap[str(colorMap[i])] = i

    #cv2.imshow("image",image)
    img = cv2.resize(image,(256,256))
    #cv2.imshow("img",img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()'''
    labeled_point_cloud = (labeled_point_cloud / np.array([col,row,1,1]))* np.array([256,256,1,1])
    mask = np.zeros((256,256))
    '''for i in range(256):
        for j in range(256):
            mask[i][j] = get_nearest_label(list(img[i][j]), colorMap)
            #print(img[i][j], ' ', mask[i][j])'''
            
    for point in labeled_point_cloud:
        mask[int(point[1])][int(point[0])] = int(point[3]+1)
    
    return mask.reshape((1,256*256))

def applyTresholding(threshold, labeled_point_cloud):
    to_process = []
    i = 0
    current_color = 0
    while current_color < 25:
        for i in range(1024):
            if labeled_point_cloud[i][3] == current_color:
                to_process.append(labeled_point_cloud[i])
        
        while len(to_process) > 0:
            point = to_process.pop(0)
            for j in range(1024):
                if labeled_point_cloud[j][3] == current_color or labeled_point_cloud[j][3] < current_color:
                    continue
                x1 = point[0]
                y1 = point[1]
                x2 = labeled_point_cloud[j][0]
                y2 = labeled_point_cloud[j][1]
                if euclidean(x1,y1,x2,y2) < threshold:
                    labeled_point_cloud[j][3] = current_color
                    to_process.append(labeled_point_cloud[j])
                    
        current_color += 1

    '''for i in range(1024):
        #labeled_point_cloud = labeled_point_cloud[np.argsort(labeled_point_cloud[:,3])]
        for j in range(1024):
            if labeled_point_cloud[i][3] == labeled_point_cloud[j][3]:
                continue
            x1 = labeled_point_cloud[i][0]
            y1 = labeled_point_cloud[i][1]
            x2 = labeled_point_cloud[j][0]
            y2 = labeled_point_cloud[j][1]
            if euclidean(x1,y1,x2,y2) < threshold:
                labeled_point_cloud[j][3] = min(labeled_point_cloud[i][3], labeled_point_cloud[j][3])'''

    labeled_point_cloud = labeled_point_cloud[np.argsort(labeled_point_cloud[:,3])]
    color = -1
    prev = -1
    for points in labeled_point_cloud:
        #print(points[3], ' ', prev)
        if points[3] != prev:
            color += 1
            prev = points[3]
        #print("setting color to ",color)
        points[3] = color
    return labeled_point_cloud

def draw_grid(img, size, i,line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA):
    x_step = int(size[0]/5)
    y_step = int(size[1]/5)
    x = 0 #int(size[0]/5)
    y = 0 #int(size[1]/5)
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += x_step

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += y_step
        
    #cv2.imwrite("../track/labeled_point_cloud/grid_"+str(i)+".png",img)
    
    return img

def check_point_lies_in_cell(point, grid):
    x = point[0]
    y = point[1]
    dist = np.inf
    cell_no = -1
    for i in range(25):
        if x >= grid[i][0] and x <= grid[i][2] and  y >= grid[i][1] and y <= grid[i][3]:
            cell_no = i
            break
    return cell_no

def get_points_image_size(image):
    image_height, image_width, _ = image.shape
    size = [image_width, image_height]
    return size

def color_stroke_cells(image, grid, point_cloud):        
    color = 0
    for point in point_cloud:
        if point[0] == 0 and point[1] == 0:
            break
        cell_no = check_point_lies_in_cell(point,grid)
        if cell_no == -1:
            continue
        if grid[cell_no][4] == -1:
            grid[cell_no][4] = color
            grid[cell_no][5] = 0
            center_x1 = int((grid[cell_no][2]+grid[cell_no][0])/2)
            center_y1 = int((grid[cell_no][3]+grid[cell_no][1])/2)
            cv2.putText(img=img, text=str(color), org=(center_x1, center_y1), 
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(255, 0, 0),thickness=1)
            color += 1
            
    return grid
            
def color_uncolored_cells(image, grid, size, img_number):
    for i in range(25):
        dist = np.inf
        for j in range(25):
            if i == j:
                continue
            if grid[i][4] != -1:
                break
            if grid[j][4] != -1 and grid[j][5] != -1:
                center_x1 = (grid[i][2]+grid[i][0])/2
                center_y1 = (grid[i][3]+grid[i][1])/2
                center_x2 = (grid[j][2]+grid[j][0])/2
                center_y2 = (grid[j][3]+grid[j][1])/2
                euclid_dist = euclidean(center_x1,center_y1,center_x2,center_y2)
                if euclid_dist < dist:
                    dist = euclid_dist
                    color = grid[j][4]
        
        c = (0,0,255)
        if grid[i][4] == -1:
            grid[i][4] = color
            c = (255,0,0)
        center_x1 = int((grid[i][2]+grid[i][0])/2)
        center_y1 = int((grid[i][3]+grid[i][1])/2)
        cv2.putText(img=img, text=str(grid[i][4]), org=(center_x1, center_y1), 
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=c,thickness=1)
                    
    draw_grid(img,size,img_number)
    return grid

def color_grid_cells(img,point_cloud, size, img_number):
    grid = []
    x = 0
    y = 0
    col = int(size[0]/5)
    row = int(size[1]/5)
    for i in range(5):
        x = 0
        for j in range(5):
            grid.append([x , y , x+col-1, y+row-1, -1, -1])            
            x += col
        y += row
        
    grid = color_stroke_cells(img,grid,point_cloud)
    #print(grid)
    grid = color_uncolored_cells(img, grid, size, img_number)
    #print(grid)
    return grid
    
def get_labeled_point_cloud(point_cloud,grid):
    for point in point_cloud:
        cell_no = check_point_lies_in_cell(point,grid)
        point[3] = grid[cell_no][4]
    #print(point_cloud)
    return point_cloud


labeled_point_cloud = np.zeros((1,1024*4))
f = open("../val_stroke_ann.txt",'r')
lines = f.readlines()
final_mask = np.zeros((1,256*256))
for i,line in enumerate(lines[:15]):
    #print(line)
    file_name = line.split()[0].strip()
    print("stroke/"+file_name)
    img = cv2.imread("../stroke/"+file_name)
    size = get_points_image_size(img)
    img = draw_grid(img,size, i)
    stroke_point_cloud = np.load("points_stroke_train.npy")[i].reshape((5120,6)) #* ([1 , -1 , 1, 1, 1 , 1])
    image_point_cloud = np.load("sample_points_stroke_1024_train.npy")[i].reshape((1024,3)) #* ([1 , -1 , 1])
    image = get_points_image(file_name)
    #image_point_cloud = image_point_cloud.reshape((1024,3))
    temp = np.zeros((1024,1))
    image_point_cloud = np.append(image_point_cloud,temp,axis=1)
    print(image_point_cloud.shape)
    grid = color_grid_cells(img,stroke_point_cloud,size,i)
    labeled_points = get_labeled_point_cloud(image_point_cloud, grid)
    labeled_points = labeled_points[np.argsort(labeled_points[:,0])]
    labeled_points = applyTresholding(10,labeled_points)
    #black_image = np.zeros((size[1],size[0]))
    mask = color_black_image(image,labeled_points,size[1],size[0])
    #labeled_points = labeled_points.reshape((1,1024*4))
    #labeled_point_cloud = np.append(labeled_point_cloud, labeled_points, axis=0)
    final_mask = np.append(final_mask, mask, axis=0)
np.save("labeled_mask_try.npy",final_mask[1:]) 