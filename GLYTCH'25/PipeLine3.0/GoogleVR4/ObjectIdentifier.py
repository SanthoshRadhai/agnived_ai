import os
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator


def run_object_detection(image_path, labels, output_dir="detected_crops", score_threshold=0.4, text_threshold=0.3):
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = Accelerator().device
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    text_labels = [labels]
    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=score_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )
    result = results[0]
    crops = []
    for idx, (box, score, label) in enumerate(zip(result["boxes"], result["scores"], result["labels"])):
        box = [round(x, 2) for x in box.tolist()]
        xmin, ymin, xmax, ymax = map(int, box)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(W, xmax)
        ymax = min(H, ymax)
        if xmax - xmin < 20 or ymax - ymin < 20:
            continue  # skip tiny crops
        crop = image.crop((xmin, ymin, xmax, ymax))
        crop_path = os.path.join(output_dir, f"crop_{idx:03d}_{label}_s{score:.2f}.jpg")
        crop.save(crop_path)
        crops.append({
            "label": label,
            "score": float(score),
            "box": box,
            "crop_path": crop_path
        })
    return {
        "image_path": image_path,
        "labels": labels,
        "crops": crops,
        "num_crops": len(crops),
        "output_dir": output_dir
    }