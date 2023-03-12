from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

MODEL = tf.keras.models.load_model("../models/2")
CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]

app = FastAPI()

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

# Takes image and returns numpy array
def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/")
async def root():
    return "Welcome to Potato Disease Classification"

@app.post("/predict")
async def predict(
        file: UploadFile = File()
):
    # Convert file to numpy array
    image = read_file_as_image(await file.read())

    # Convert image to batch as predict() requires batch of images
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)




#docker run -p 8501:8501 --name tfserving_resnet --mount type=bind,source=/tmp/resnet,target=/models/resnet -e MODEL_NAME=resnet -t emacski/tensorflow-serving:latest-linux_arm64