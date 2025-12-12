from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import torch

model_id = "juppy44/plant-identification-2m-vit-b"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

url = "./tree-1.jpg"
image = Image.open(url)

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

pred = logits.softmax(dim=-1)[0]
topk = torch.topk(pred, k=5)

for prob, idx in zip(topk.values, topk.indices):
    label = model.config.id2label[idx.item()]
    print(f"{label}: {prob.item():.4f}")
