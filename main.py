from fastapi import FastAPI, UploadFile, File
from app.model.predict import predict_image
import shutil
import json

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        path = f"input_images/temp_{file.filename}"
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_image(path)
        return {"prediction": result}
    except:
        return { "status": "500", "message": "Internal Server Error" }

