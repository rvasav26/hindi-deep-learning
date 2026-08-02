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
import requests

canvas = np.zeros((640, 640, 1), np.uint8)
canvas.fill(255)


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

    # Send the drawing to the inference API.
    image_to_send = np.invert(canvas)

    _, buffer = cv2.imencode(".png", image_to_send)

    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("image.png", buffer.tobytes(), "image/png")},
    )

    response.raise_for_status()

    result = response.json()
    finalPred = result["char"]

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
