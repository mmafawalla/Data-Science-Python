from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = "potato-leaf-classification-bucket"

CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]

model = None

# Dowbload model from storage
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def predict(request):
    # Download model only once
    global model
    if model is not None:
        # Download model
        download_blob(
            BUCKET_NAME,
            "models/potato_leaf_classification.h5",
            "/tmp/potato_leaf_classification.h5"
        )
        model = tf.keras.load_model("/tmp/potato_leaf_classification.h5")
    # Prediction image we upload
    image = request.files["file"]

    # Resize the image to our standard size
    image = np.array(Image.open(image).convert("RGB").resize((256, 256)))
    # Scale image values to between 0 and 1
    image = image/255

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print(predictions)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round((np.max(predictions[0])) * 100, 2)

    return {
        "class": predicted_class,
        "confidence": confidence
    }

# gcloud functions deploy predict --runtime python38 --trigger-http --memory 512 --project useful-maxim-379704
