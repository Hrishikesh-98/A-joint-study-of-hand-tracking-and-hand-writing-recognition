import cv2
import mediapipe as mp
import math
import numpy as np
import csv
import pandas as pd
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

push = False
started = False
ended = False
start = 0
end  = 0
i = 0
#i = 0
indices = np.zeros((1,2))
def click(event, x, y, flags, param):
    global push
    global start
    global end
    global i
    global started
    global ended
    global indices
    if event == cv2.EVENT_RBUTTONDOWN:
      if push:
        push = False
        started = False
        ended = True
        end = i
        index = np.array([start, end])
        text = "hi"
        with open('words_str.txt','a') as files:
          files.write(text+'\n')
        with open('indices_str.csv','a') as file:
          write = csv.writer(file)
          write.writerow(index)
        print("ended")
        print(index.reshape((1,2)).shape)
        print(indices)
        indices = np.append(indices,index.reshape((1,2)),axis=0)
        print(indices)
    if event == cv2.EVENT_LBUTTONDOWN:
      push = True
      started = True
      start = i
      print("started")
        

images = []
keypoints = []
cap = cv2.VideoCapture(0)
cv2.namedWindow("MediaPipe Hands")
cv2.setMouseCallback("MediaPipe Hands", click)
vid = 0
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    #print(len(image))
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    #image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    img = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    land = []
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        for j in range(21):
          joint = [hand_landmarks.landmark[j].x,
                   hand_landmarks.landmark[j].y,
                   hand_landmarks.landmark[j].z]
          land.append(joint)
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if push:
          keypoints.append(land)
          i= i+1
    cv2.imshow('MediaPipe Hands', image)
    if started:
      images.append(img)
      print("appending")
    if ended:
      image_height, image_width, _ = img.shape
      size = (image_width,image_height)
      out = cv2.VideoWriter(str(vid)+'_large_test.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
      for k in range(len(images)):
          out.write(images[k])
          print('writting')      
          started = False
          ended = False
      out.release()
      images.clear()
      vid += 1
    if cv2.waitKey(10) == ord('q'):
      break



'''path = './keypoints.npy'
existing = np.load(path,allow_pickle=True)
keypoint = np.vstack((existing,np.array(keypoints)))
# Save features

print(len(keypoints))
print(np.array(keypoints).shape)
np.save('./keypoints.npy',np.array(keypoint))

'''

print("Saving keypoints file")

arr = np.array(keypoints)

print("indices ",indices[1:].shape)
np.save("indces_str_test.npy",indices[1:])
#keys = np.load("keypoints.npy")
#arr = np.append(keys,arr.reshape((arr.shape[0],63)),axis=0)
print(arr.shape)
np.save("keypoints_str_test.npy",arr)
    
print('keyponits file saved')

