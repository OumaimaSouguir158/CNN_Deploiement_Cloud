"""
API d'inférence Cloud pour le modèle CNN (transfer learning MobileNetV2 /
CIFAR-10). Conçue pour être déployée sur un service Cloud avec offre
gratuite (ex. Render, Railway, Google Cloud Run, Hugging Face Spaces).
"""

import io
import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

CLASSES = ["avion", "automobile", "oiseau", "chat", "cerf",
           "chien", "grenouille", "cheval", "bateau", "camion"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "cnn_cifar10_saved_model")

app = FastAPI(
    title="API de classification d'images (CNN + transfer learning)",
    description="Reçoit une image et retourne la classe prédite parmi 10 catégories CIFAR-10.",
    version="1.0.0",
)

model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    array = np.array(image)
    return np.expand_dims(array, axis=0)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    contents = await file.read()
    x = preprocess_image(contents)
    predictions = model.predict(x)[0]

    top_idx = int(np.argmax(predictions))
    return {
        "classe_predite": CLASSES[top_idx],
        "confiance": round(float(predictions[top_idx]), 4),
        "toutes_probabilites": {
            CLASSES[i]: round(float(p), 4) for i, p in enumerate(predictions)
        },
    }
