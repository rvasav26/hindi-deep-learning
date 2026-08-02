import cv2

from .inference import predict

image = cv2.imread(
    "/Users/rhushilvasavada/Desktop/Other/Rhushil_Software_Dev/Machine Learning/hindi-deep-learning/data/DevanagariHandwrittenCharacterDataset/Test/character_11_taamatar/190.png",
    cv2.IMREAD_GRAYSCALE,
)

print(predict(image))
