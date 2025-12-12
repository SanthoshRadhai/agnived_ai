#!/usr/bin/env python3

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch

def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return Image.open(response.raw).convert("RGB")

def main():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    print("[*] Downloading image...")
    image = load_image_from_url(url)

    print("[*] Loading processor and fine-tuned ViT model...")
    model_name = "bryanzhou008/vit-base-patch16-224-in21k-finetuned-inaturalist"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("[*] Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]

    print("\n✅ PREDICTION RESULT")
    print("-------------------")
    print(f"Predicted class index: {predicted_class_idx}")
    print(f"Predicted label: {predicted_label}")

if __name__ == "__main__":
    main()
