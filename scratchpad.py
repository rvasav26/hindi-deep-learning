# Author: Rhushil Vasavada
# Deep Learning Hindi Scratchpad
# Description: This program enables a user to draw on a digital scratchpad where their
# Hindi handwriting can be analyzed and assessed by a custom trained PyTorch Convolutional
# Neural Network (CNN) in real time. Users can draw with their mouse and the CNN will recognize
# the handwriting nearly instantaneously.

# import necessary libraries
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import onnxruntime as ort
from devanagari_model import DEVANAGARI_CLASSES

canvas = np.zeros((640, 640, 1), np.uint8)
canvas.fill(255)

# since the model outputs English-indexed classes representing the Hindi letters, we use
# UTF-8 encoding to represent the corresponding letters in their original Hindi form:
classList = DEVANAGARI_CLASSES

# load custom convolutional neural network model
session = ort.InferenceSession(
    "hindi_cnn_int8.onnx",  # or hindi_cnn.onnx
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name

# load font once, outside the loop (your original reloaded this every single frame)
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/DevanagariMT.ttc", 200)

x = 0
y = 0
drawing = False


# function to enable user drawing based on mouse movement
def mouseDraw(event, current_x, current_y, flags, params):
    global x, y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        x = current_x
        y = current_y
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (current_x, current_y), (x, y), 0, thickness=30)
            x, y = current_x, current_y


# enable drawing for the canvas screen
cv2.imshow("Scratchpad", canvas)
cv2.setMouseCallback("Scratchpad", mouseDraw)

while True:
    # create a new canvas array where the predicted character will be displayed
    imagePIL = np.zeros((640, 640, 1), np.uint8)
    imagePIL.fill(255)
    imagePIL = cv2.cvtColor(imagePIL, cv2.COLOR_GRAY2RGB)

    # manipulate array to enable program to write in Devanagari script
    pil_image = Image.fromarray(imagePIL)
    draw = ImageDraw.Draw(pil_image)

    # perform matrix transformations to prepare user's handwriting to be fed into model
    # for prediction
    imgPred = cv2.resize(canvas, (32, 32))
    imgPred = np.invert(np.array([imgPred]))
    imgPred = (
        imgPred.reshape(1, 1, 32, 32).astype(np.float32) / 255
    )  # PyTorch wants NCHW, not NHWC

    # run the model on the transformed matrix containing the handwriting as a tensor
    prediction = session.run(
        None,
        {input_name: imgPred},
    )[0]

    # store the prediction (Devanagari character with highest match with user's writing)
    finalPred = classList[np.argmax(prediction, axis=1)[0]]

    # draw the prediction on the output window
    draw.text((250, 200), str(finalPred), font=font, fill="black")

    # convert the pil_image (separate format specifically for writing unique characters)
    # into standard image
    letterOut = np.asarray(pil_image)

    # clear scratchpad if user enters "w" (reset)
    if cv2.waitKey(1) & 0xFF == ord("w"):
        canvas = np.zeros((640, 640, 1), np.uint8)
        canvas.fill(255)

    # display all necessary windows for output
    cv2.imshow("Prediction", letterOut)
    cv2.moveWindow("Prediction", 640, -200)
    cv2.imshow("Scratchpad", canvas)

    cv2.waitKey(1)

# destroy all windows once program has terminated
cv2.destroyAllWindows()
