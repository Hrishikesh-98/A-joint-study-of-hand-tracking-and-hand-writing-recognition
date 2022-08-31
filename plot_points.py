import matplotlib.pyplot as plt
import numpy as np
import csv

def get_color_value(a):
    if a == 0:
        return "black"
    if a == 1:
        return "red"
    if a == 2:
        return "blue"
    if a == 3:
        return "orange"
    if a == 4:
        return "green"
    if a == 5:
        return "mediumvioletred"
    if a == 6:
        return "yellow"
    if a == 7:
        return "cyan"
    if a == 8:
        return "lawngreen"
    if a == 9:
        return "hotpink"
    if a == 10:
        return "lime"
    if a == 11:
        return "magenta"
    if a == 12:
        return "darkcyan"
    if a == 13:
        return "peru"
    if a == 14:
        return "gold"
    if a == 15:
        return "coral"
    if a == 16:
        return "lightgreen"
    if a == 17:
        return "steelblue"
    if a == 18:
        return "crimson"
    if a == 19:
        return "royalblue"
    if a == 20:
        return "darkred"
    if a == 21:
        return "darkviolet"
    if a == 22:
        return "goldenrod"
    if a == 23:
        return "orangered"
    if a == 24:
        return "mediumvioletred"

def load_csv(file1):
    with open(file1, 'r') as f:
        reader = csv.reader(f)
        x = list(reader)

    new_x = []
    for row in x:
        nwrow = []
        for r in row:
            nwrow.append(np.array(r[1:-1].split()))
        new_x.append(nwrow)

    return np.float64(new_x)

def BezierPoint(a):
    global ind
    index = ind
    global l
    
    for i in range(l):
        color = get_color_value(x[index][i][2])
        plt.plot(x[index][i][0],-x[index][i][1],'*', color=color)
    return

def Show(a):
    BezierPoint(a)
    return


if __name__ == '__main__':
    x=[]
    y=[]
    clist = list()
    l = 1024
    x = np.load('labeled_point_stroke_train.npy')
    x = x.reshape(x.shape[0],l,3)
    print
    l = x.shape[1]
    print(x.shape)
    for i in range(10,15):
        print(x[i])
        ind = i
        fig = plt.figure()
        ax = fig.add_subplot(111)
        Show(0)
        plt.show()