import cv2
import mediapipe as mp
import math
import os
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
    
def create_video(vid,overlay,mp):

    #file_list = list()
    #file_list.append('./data/hand.jpg')
    cap= cv2.VideoCapture(vid+'_large_test.avi')
    images = []
    size = ()
    keys_original = np.load("keypoints_str_test.npy")
    keys_our = np.load("keys_MSE_adam_best.npy")
    index = np.load("indces_str_test.npy")[0:]
    keys_our = keys_our.reshape((keys_our.shape[0],21,3))
    keys_original = keys_original.reshape((keys_original.shape[0],21,3))
    ind = int(index[int(vid)][0]) #-int(index[25][0])
    i = 0
    results = None
    print(int(index[int(vid)][1]))
    image = cv2.imread("img.jpg")
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print(not results.multi_hand_landmarks)
    universal_landmark = None
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while(cap.isOpened()):
            ret, image = cap.read()
            # Convert the BGR image to RGB before processing.
            if not ret:
                break
            if ret:
                '''results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                #print(type(results.multi_hand_landmarks))
                # Print handedness and draw hand landmarks on the image.
                #print('Handedness:', results.multi_handedness)
                if not results.multi_hand_landmarks and not universal_landmark:
                  print("index ",i)
                  continue
                if universal_landmark == None:
                    print("inside None")
                    universal_landmark =  results.multi_hand_landmarks
                    
                if universal_landmark is not None:
                    print("inside not None")
                    results.multi_hand_landmarks = universal_landmark'''
                image_height, image_width, _ = image.shape
                annotated_image = image.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                  #print('hand_landmarks:', type(hand_landmarks), '\n ', type(hand_landmarks.landmark))
                  '''print(
                      f'Index finger tip coordinates: (',
                      f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                      f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                  )'''
                  if overlay or mp:
                      for j in range(21):
                        hand_landmarks.landmark[j].x = keys_original[ind][j][0]
                        hand_landmarks.landmark[j].y = keys_original[ind][j][1]
                        hand_landmarks.landmark[j].z = keys_original[ind][j][2]
                      mp_drawing.draw_landmarks(
                          annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                          #,mp_drawing_styles.get_default_hand_landmarks_style(),
                          #mp_drawing_styles.get_default_hand_connections_style())
                          )
                      
                if not mp:
                    for hand_landmarks in results.multi_hand_landmarks:
                      #print('hand_landmarks:', type(hand_landmarks), '\n ', type(hand_landmarks.landmark))
                      '''print(
                          f'Index finger tip coordinates: (',
                          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                      )'''
                      for j in range(21):
                        hand_landmarks.landmark[j].x = keys_our[ind][j][0]
                        hand_landmarks.landmark[j].y = keys_our[ind][j][1]
                        hand_landmarks.landmark[j].z = keys_our[ind][j][2]
                      mp_drawing.draw_landmarks(
                          annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                          ,mp_drawing_styles.get_default_hand_landmarks_style(),
                          mp_drawing_styles.get_default_hand_connections_style()
                          )
                #cv2.imwrite(
                #    './results/' + str(idx) + '.png', cv2.flip(annotated_image, 1))
                size = (image_width,image_height)
                images.append(annotated_image)
                cv2.imshow("img",annotated_image)
                cv2.waitKey(5)
                cv2.destroyAllWindows()
                #print(size)
                ind += 1
            i += 1
    print()
    print(size)
    if overlay:
        name = 'overlaid_output_'
    elif mp:
        name = './stroke_frames/test/'
    else:
        name = 'model_output_'
    os.makedirs("./stroke_frames/test/"+str(int(vid)+101)+"/",exist_ok=True)
    print("./stroke_frames/test/"+str(int(vid)+101)+"/")
    #out = cv2.VideoWriter(name+str(vid)+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    for i in range(len(images)):
        cv2.imwrite(name+str(int(vid)+101)+'/' + str(i)+'.png',images[i]) #out.write(images[i])
        print('writting')
    #out.release()


for i in range(31,32):
    create_video(str(i),0,1)

# For static images:
'''
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(file_list):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    print(file)
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    middle_finger_tip = []
    wrist = []
    print(results.multi_hand_landmarks[0].landmark[0].x)
    for i,hand_landmarks in enumerate(results.multi_hand_landmarks):
      #print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    wrist_x = results.multi_hand_landmarks[0].landmark[0].x #* image_width
    wrist_y = results.multi_hand_landmarks[0].landmark[0].y #* image_height
    mft_x = results.multi_hand_landmarks[0].landmark[9].x #* image_width
    mft_y = results.multi_hand_landmarks[0].landmark[9].y #* image_height
    euc_dist = math.sqrt( (mft_x - wrist_x)**2 +  (mft_y - wrist_y)**2 )
    print(euc_dist)
    cv2.imwrite(
        './results/' + str(idx) + '.png', cv2.flip(annotated_image, 1))
        
'''        

'''
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    print(len(image))
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    #image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
'''

'''
import cv2
import mediapipe as mp
import time
class handDetector():
    def __init__(self, mode = False, maxHands = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
'''