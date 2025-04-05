# Author: Rhushil Vasavada
# Deep Learning Hindi Airpad
# Description: This program enables a user to have a digital airpad where their
# Hindi handwriting can be analyzed and assessed by a custom-trained TensorFlow Convolutional 
# Neural Network (CNN) in real-time. Users can draw in the air with their pointer finger
# and the program will detect the user's drawing. A user is drawing when their mouth is closed,
# and the user is not drawing when their mouth is open. The drawing is fed into the CNN and 
# the prediction is displayed.

# -*- coding: utf-8 -*-

# import necessary libraries
import mediapipe as mp
import cv2
import numpy as np
import hand_tracking_module as htm
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from keras.models import load_model

# since TensorFlow model outputs English characters representing the Hindi letters, we use
# UTF-8 encoding to represent the corresponding letters in their original Hindi form:
classList = [u'\u091E', u'\u091F', u'\u0920', u'\u0921', u'\u0922', u'\u0923', u'\u0924', u'\u0925', u'\u0926',
             u'\u0927', u'\u0915', u'\u0928', u'\u092A', u'\u092B', u'\u092c', u'\u092d', u'\u092e', u'\u092f',
             u'\u0930', u'\u0932', u'\u0935', u'\u0916', u'\u0936', u'\u0937', u'\u0938', u'\u0939',
             '''No UTF-8 encoding for the following Hindi Characters:''' 'क्ष ', 'त्र ', 'ज्ञ ',
             u'\u0917', u'\u0918', u'\u0919', u'\u091a', u'\u091b', u'\u091c', u'\u091d', u'\u0966', u'\u0967',
             u'\u0968', u'\u0969', u'\u096a', u'\u096b', u'\u096c', u'\u096d', u'\u096e', u'\u096f']

model = load_model('hindi_cnn_weights_tf2.h5')

# start recording video
cap = cv2.VideoCapture(0)

# variables used for parameters later on
myPoints = []
count = 0
airpadPenSize = 12
airpadLineSize = 21

# collect MediaPipe modules for detecting hands and detecting face landmarks from a face mesh
detector = htm.handDetector(detectionCon=0.7)
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=5)
drawSpec = mpDraw.DrawingSpec(thickness=2, circle_radius=2, color=(255, 255, 255))

# used to draw user's handwriting
imgBlank = np.zeros((320, 320, 3), dtype=np.uint8)
imgBlank.fill(255)

while True:
    # create a new canvas array where the predicted character will be displayed
    imagePIL = np.zeros((320, 320, 1), np.uint8)

    # make canvas white
    imagePIL.fill(255)
    imagePIL = cv2.cvtColor(imagePIL, cv2.COLOR_GRAY2RGB)

    # load fonts and manipulate array to enable program to write in Devanagari script
    pil_image = Image.fromarray(imagePIL)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/DevanagariMT.ttc", 200)
    draw = ImageDraw.Draw(pil_image)

    # read camera footage from webcam
    success, img = cap.read()

    # resize webcam display
    imgDimensions = img.shape
    img = cv2.resize(img, (int(imgDimensions[1] / 1.17), int(imgDimensions[0] / 1.17)))

    # find hands and the hand landmark positions from webcam
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    # create facemesh from webcam
    success, imgFace = cap.read()
    imgRGB = cv2.cvtColor(imgFace, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        # draw each facial landmark (468 of them)
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

        # get each landmark's position
        lms = list(enumerate(faceLms.landmark))
        ih, iw, ic = img.shape

        # we are looking for the upper lip and bottom lip landmarks (namely, landmarks 15 and 13)
        x, y = int(lms[15][1].x * iw), int(lms[15][1].y * ih)
        x2, y2 = int(lms[13][1].x * iw), int(lms[13][1].y * ih)

        # draw the upper lip and lower lip landmarks in a different color, and draw a line between them
        # representing the distance
        cv2.circle(img, (x, y), 2, (0, 10, 240), 5)
        cv2.circle(img, (x2, y2), 2, (0, 10, 240), 5)
        cv2.line(img, (x, y), (x2, y2), (0, 191, 30), 3)
        
        distance = y - y2

    if distance < 20:
        # if the distance between the lips is low (mouth closed), the user should be drawing:
        if len(lmList) != 0:
            x2, y2 = lmList[8][1], lmList[8][2]
            myPoints.append([x2, y2, count])
            cv2.circle(img, (x2, y2), airpadPenSize, (255, 255, 255), cv2.FILLED)
           
    else:
        # otherwise, the user should not be drawing:
        if len(lmList) != 0:
            x2, y2 = lmList[8][1], lmList[8][2]
            cv2.circle(img, (x2, y2), airpadPenSize, (255, 255, 255), cv2.FILLED)

        myPoints.append(["S"])

    for point in myPoints:
        # draw each point in the array containing the points the user has drawn
        if point[0] != "S":
            cv2.circle(img, (point[0], point[1]), airpadPenSize, (0, 69, 255), cv2.FILLED)
            cv2.circle(imgBlank, (point[0], point[1]), airpadPenSize, (0, 0, 0), cv2.FILLED)

    for z in range(len(myPoints) - 1):
        # to make the appearance of drawing smoother, draw a line between each point of the user's
        # handwriting
        if myPoints[z + 1][0] != "S" and myPoints[z][0] != "S":
            cv2.line(img, (myPoints[z][0], myPoints[z][1]), (myPoints[z + 1][0], myPoints[z + 1][1]), (0, 69, 255),
                     airpadLineSize)
            cv2.line(imgBlank, (myPoints[z][0], myPoints[z][1]), (myPoints[z + 1][0], myPoints[z + 1][1]), (0, 0, 0), 
                     airpadLineSize)

    # rectangle to show where user should be drawing
    cv2.rectangle(img, (320, 320), (0, 0), (255, 0, 0), 2)

    # flip canvases into proper orientation (webcam reflects footage)
    img = cv2.flip(img, 1)
    imgBlank2 = cv2.flip(imgBlank, 1)

    # get detected handwriting and reshape into an array that is comaptible with model
    imgPred = cv2.cvtColor(imgBlank2, cv2.COLOR_BGR2GRAY)
    imgPred = cv2.resize(imgPred, (32, 32))
    imgPred = np.invert(np.array([imgPred]))
    imgPred = imgPred.reshape(1, 32, 32, 1) / 255

    # run the CNN on the predicted handwriting (fed as an array)
    prediction = model.predict([imgPred], verbose=0)

    # output the prediction in original Hindi form
    draw.text((100, 55), str(classList[prediction.argmax()]), font=font, fill="black")

    # change output canvas to be compatible with cv2.imshow() function
    letterOut = np.asarray(pil_image)
    letterOut = cv2.cvtColor(letterOut, cv2.COLOR_RGB2GRAY)

    # clear scratchpad if user enters "w" (reset)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        myPoints = []
        imgBlank = np.zeros((320, 320, 3), dtype=np.uint8)
        imgBlank.fill(255)

    # display all necessary windows for output
    cv2.imshow('Prediction', letterOut)
    cv2.moveWindow("Prediction", 0, 270)
    cv2.imshow('Image', img)
    cv2.imshow('Detected Writing', imgBlank2)

    cv2.waitKey(1)
  
# destroy all windows once program has terminated
cv2.destroyAllWindows()
