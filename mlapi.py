from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import tensorflow as tf
import uvicorn

model_path = os.path.join(os.getcwd(), r"model1.h5")
model = tf.keras.models.load_model(model_path, compile=False)

plant_names = [
    "Acacia mearnsii",
    "Antigonon leptopus",
    "Arundo donax",
    "Cabomba furcata",
    "Chromolaena odorata",
    "Ipomoea cairica",
    "Lantana camara",
    "Mikania micrantha",
    "Mimosa diplotricha",
    "Parthenium hysterophorus",
    "Pistia stratiotes",
    "Prosopis juliflora",
    "Salvinia molesta",
    "Senna spectabilis",
    "Sphagneticola trilobata",
    "Pontederia crassipes",
]

app = FastAPI(title="EcoAware")

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers,
)


@app.get("/")
def home():
    return "Welcome to EcoAware!"


@app.post("/api/predict")
async def predict_from_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/jpg"]:
        return {"Error": "This api can only take in images of jpeg format"}
    file = await file.read()
    image = tf.image.decode_image(file, channels=3, dtype=tf.uint8)
    image = tf.image.resize(image, (256, 256))
    image = tf.reshape(tf.constant(image), (1, 256, 256, 3))
    prediction = model.predict(image, verbose=0)
    pred_label = tf.argmax(prediction[0]).numpy()
    return {
        "Prediction": plant_names[pred_label],
        "Confidence": round(float(prediction[0][pred_label]), 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
