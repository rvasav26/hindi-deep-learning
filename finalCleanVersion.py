# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from keras.models import load_model

canvas = np.zeros((320, 320, 1), np.uint8)
canvas.fill(255)
x = 0
y = 0
drawing = False

classList = [u'\u091E', u'\u091F', u'\u0920', u'\u0921', u'\u0922', u'\u0923', u'\u0924', u'\u0925', u'\u0926',
             u'\u0927', u'\u0915', u'\u0928', u'\u092A', u'\u092B', u'\u092c', u'\u092d', u'\u092e', u'\u092f',
             u'\u0930', u'\u0932', u'\u0935', u'\u0916', u'\u0936', u'\u0937', u'\u0938', u'\u0939', 'क्ष ', 'त्र ',
             'ज्ञ ', u'\u0917', u'\u0918', u'\u0919', u'\u091a', u'\u091b', u'\u091c', u'\u091d', u'\u0966', u'\u0967',
             u'\u0968', u'\u0969', u'\u096a', u'\u096b', u'\u096c', u'\u096d', u'\u096e', u'\u096f']

model = load_model('hindi_OCR_cnn_model_tf2.h5')

def draw(event, current_x, current_y, flags, params):
    global x, y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        x = current_x
        y = current_y
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (current_x, current_y), (x, y), 0, thickness=20)
            x, y = current_x, current_y


cv2.imshow('Draw', canvas)
cv2.setMouseCallback('Draw', draw)

while True:
    imagePIL = np.zeros((320, 320, 1), np.uint8)

    imagePIL.fill(255)
    imagePIL = cv2.cvtColor(imagePIL, cv2.COLOR_GRAY2RGB)

    pil_image = Image.fromarray(imagePIL)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/DevanagariMT.ttc", 200)
    draw = ImageDraw.Draw(pil_image)

    imgPred = cv2.resize(canvas, (32, 32))
    imgPred = np.invert(np.array([imgPred]))
    imgPred = imgPred.reshape(1, 32, 32, 1) / 255

    prediction = model.predict([imgPred], verbose=0)

    finalPred = classList[prediction.argmax()]

    draw.text((100, 60), str(finalPred), font=font, fill="black")

    letterOut = np.asarray(pil_image)

    if cv2.waitKey(1) & 0xFF == ord('w'):
        canvas = np.zeros((320, 320, 1), np.uint8)
        canvas.fill(255)

    cv2.imshow('LetterOut', letterOut)
    cv2.moveWindow("LetterOut", 320, -200)
    cv2.imshow('Draw', canvas)
    cv2.waitKey(1)

cv2.destroyAllWindows()
