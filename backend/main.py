from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np

from backend.inference import predict

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_character(file: UploadFile = File(...)):
    """
    Receive an image and return predicted Devanagari character.
    """

    image = Image.open(file.file)

    image = np.array(image)

    result = predict(image)

    return result
