#!/usr/bin/env python3



Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-base-patch16-224-in21k and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.        
[*] Using device: cuda

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return Image.open(response.raw).convert("RGB")


def main():
    # 1. Image URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # 2. Load image
    print("[*] Downloading image...")
    image = load_image_from_url(url)

    # 3. Load processor and classification model
    print("[*] Loading ViT classification model...")
    processor = ViTImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )

    # 4. Device setup (CPU / GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[*] Using device: {device}")

    # 5. Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 6. Forward pass (prediction)
    print("[*] Running prediction...")
    with torch.no_grad():
        outputs = model(**inputs)

    # 7. Get predicted class
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # 8. Decode label
    predicted_label = model.config.id2label[predicted_class_idx]

    print("\n✅ PREDICTION RESULT")
    print("-------------------")
    print(f"Predicted Class Index : {predicted_class_idx}")
    print(f"Predicted Label       : {predicted_label}")


if __name__ == "__main__":
    main()
