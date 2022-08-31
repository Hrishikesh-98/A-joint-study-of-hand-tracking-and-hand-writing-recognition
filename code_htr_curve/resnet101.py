import cv2
import math
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import torch
import torch.nn as nn
import numpy as np

#file_list = list()
#file_list.append('./data/hand.jpg')

import torch

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = models.resnet101(pretrained=True)
model = nn.Sequential(*list(model.children())[:-4])
#model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model.eval()

features = np.zeros((1,512*32*32))
f = open("hwnet/ann/test_new_ann.txt",'r')
lines = f.readline()
for i,line in enumerate(lines):
    filename = line.split()[0]
    input_image = cv2.imread("../../../extra/data/hrishikesh/words_dataset/"+filename)
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(img)
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch).permute(0,2,3,1).reshape(-1,512*32*32)
        print(i,' ',output.shape)
    features = np.append(features,output,axis=0)
            
np.save("feature_image_test.npy",features[1:])
