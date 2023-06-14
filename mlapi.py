from fastapi import FastAPI, File, UploadFile
import uvicorn
import tensorflow as tf
from PIL import Image
import io

app = FastAPI(title="ECOAWARE")

model_path = (
    r"C:\Users\zuraj\Downloads\MiniProjectNew\Programs\Saved models\model1_2.h5"
)
model = tf.keras.models.load_model(model_path)

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


@app.get("/")
def home():
    return "Welcome to the site!"


@app.post("/api/predict")
async def predict_from_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/jpg"]:
        return {"Error": "This api can only take in images of jpeg format"}
    file = await file.read()
    image = tf.image.decode_image(file, channels=3, dtype=tf.uint8)
    image = tf.image.resize(image, (256, 256))
    image = tf.reshape(tf.constant(image), (1, 256, 256, 3))
    prediction = model.predict(image)
    pred_label = tf.argmax(prediction[0]).numpy()
    return "Prediction is " + plant_names[pred_label]


if __name__ == "__main__":
    uvicorn.run(app=app, host="127.0.0.1", port=8000)
