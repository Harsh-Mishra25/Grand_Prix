from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

MODEL_NAME = "openai/clip-vit-base-patch32"

# Load once at startup
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Candidate labels — phrase them descriptively, CLIP responds better to full sentences
LABELS = {
    "dry": "a photo of a dry racetrack with no water on the surface",
    "damp": "a photo of a slightly damp racetrack with a wet sheen but no puddles",
    "wet": "a photo of a wet racetrack with visible puddles and standing water",
    "drying": "a photo of a racetrack that is drying, with patchy wet and dry areas",
}

def classify_image(image: Image.Image):
    label_names = list(LABELS.keys())
    label_texts = list(LABELS.values())

    inputs = processor(
        text=label_texts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # image-text similarity scores
        probs = logits_per_image.softmax(dim=1)[0]   # convert to probabilities

    scores = {label_names[i]: float(probs[i]) for i in range(len(label_names))}
    best_label = max(scores, key=scores.get)

    return {
        "label": best_label,
        "confidence": scores[best_label],
        "all_scores": scores
    }