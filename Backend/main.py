from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from classifier import classify_image
from trend import add_reading, get_trend

app = FastAPI()

# Allow your frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this later for production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = classify_image(image)
    add_reading(result["label"])

    trend_info = get_trend()

    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "all_scores": result["all_scores"],
        "trend": trend_info["direction"],
        "tire_suggestion": trend_info["suggestion"],
        "history": trend_info["history"]
    }

@app.get("/")
async def root():
    return {"status": "GrandPrix backend running"}