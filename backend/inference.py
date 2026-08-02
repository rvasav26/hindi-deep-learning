from __future__ import annotations

from pathlib import Path
from PIL import Image

import numpy as np
import onnxruntime as ort

from model.classes import DEVANAGARI_CLASSES

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "hindi_cnn.onnx"

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name


def preprocess(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = Image.fromarray(image).convert("L")
    else:
        image = Image.fromarray(image)

    image = image.resize((32, 32))

    image = np.array(image)

    image = image.astype(np.float32) / 255.0

    image = image.reshape(1, 1, 32, 32)

    return image


def predict(image: np.ndarray) -> dict:
    """
    Predict the handwritten Devanagari character.

    Parameters
    ----------
    image
        Grayscale or RGB image.

    Returns
    -------
    dict
        {
            "char": "...",
            "confidence": 0.997,
            "index": 10
        }
    """

    input_tensor = preprocess(image)

    logits, embedding = session.run(None, {input_name: input_tensor})

    logits = logits[0]  # remove batch dimension

    predicted_index = int(np.argmax(logits))

    # Softmax for confidence
    exp = np.exp(logits - np.max(logits))
    probabilities = exp / exp.sum()

    confidence = float(probabilities[predicted_index])

    return {
        "char": DEVANAGARI_CLASSES[predicted_index],
        "confidence": confidence,
        "index": predicted_index,
        "embedding": embedding[0].tolist(),
    }
