from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

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

# Load TensorFlow model
MODEL = tf.keras.models.load_model("../saved_models/1")

# Class labels
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Health Check
@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

# Convert uploaded file to numpy image
def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")  # Ensure 3 channels
    image = image.resize((256, 256))  # Resize if your model expects this size
    return np.array(image)

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = read_file_as_image(image_data)
    img_batch = np.expand_dims(image, axis=0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence
    }

# Run the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
