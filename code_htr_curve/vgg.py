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

model = models.vgg16(pretrained=True)
new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
model.classifier = new_classifier
#model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model.eval()
features = np.zeros((1,4096))
#print(np.load("keypoints_all.npy").shape)
count = 1
print(features.shape)
for i in range(0,38):
    cap= cv2.VideoCapture("../../../extra/data/hrishikesh/vids/"+str(i)+'.avi')
    images = []
    size = ()
    print(model)
    while(cap.isOpened()):
        ret, input_image = cap.read()
        if not ret:
            break
        if ret:
            img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = Image.fromarray(img)
            preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
                output = model(input_batch)
                print(i+24,' ',count , ' ',output.detach().cpu().numpy().shape)
            features = np.append(features,output.detach().cpu().numpy(),axis=0)
            count +=1
            
print(features[1:].shape)
np.save("../../../extra/data/hrishikesh/feature_image_all.npy",features[1:])
