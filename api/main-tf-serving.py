from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = FastAPI()

# CORS Settings
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TensorFlow Serving Endpoint
endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

# Label Names
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Health Check Route
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

# Convert bytes to NumPy Image
def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")  # ensure 3 channels
    image = image.resize((256, 256))  # Optional: resize if model requires it
    return np.array(image)

# Predict Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = read_file_as_image(image_data)
    img_batch = np.expand_dims(image, axis=0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {
        "class": predicted_class,
        "confidence": confidence
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
