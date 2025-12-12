from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
import torch
import torchvision.models as models

print("=" * 60)
print("DOWNLOADING PYTORCH-WILDLIFE MODELS")
print("=" * 60)

# 1. Download MegaDetector v6 (Detection only - animal/person/vehicle)
print("\n📥 Downloading MegaDetector v6...")
detector = pw_detection.MegaDetectorV6(
    device="cuda" if torch.cuda.is_available() else "cpu",
    pretrained=True,
    version="MDV6-yolov10-e" #use global variable for this later on to switch to c for optimal computing.
)
print(f"✓ MegaDetector v6 loaded successfully on: {detector.device if hasattr(detector, 'device') else 'unknown device'}")

print("\n" + "=" * 60)
print("DOWNLOADING SPECIES CLASSIFICATION MODELS")
print("=" * 60)

print("\nAvailable classification models in PytorchWildlife:")
print(dir(pw_classification))

classification_models = {
    "AI4GSnapshotSerengeti": {
        "desc": "Snapshot Serengeti - 48 African species, camera trap trained",
        "class": pw_classification.AI4GSnapshotSerengeti
    },
    "Deepfaune": {
        "desc": "European wildlife - 20+ mammals/birds including wild boar",
        "class": pw_classification.Deepfaune
    },
    "AI4GAmazonRainforest": {
        "desc": "Amazon species (tropical/rainforest animals)",
        "class": pw_classification.AI4GAmazonRainforest
    }
}

device = "cuda" if torch.cuda.is_available() else "cpu"

for model_name, info in classification_models.items():
    print(f"\n📥 Downloading {model_name}...")
    print(f"   Purpose: {info['desc']}")
    try:
        classifier = info['class'](device=device, pretrained=True)
        print(f"✓ {model_name} loaded successfully")
    except Exception as e:
        print(f"⚠️  Error loading {model_name}: {e}")

print("\n" + "=" * 60)
print("ALL MODELS DOWNLOADED")
print("=" * 60)
print(f"\nModels cached in: ~/.cache/torch/hub/")
print(f"Device: {device.upper()}")
print(f"CUDA available: {torch.cuda.is_available()}")

print("\n" + "=" * 60)
print("WORKFLOW FOR WILDLIFE IDENTIFICATION:")
print("=" * 60)
print("1. MegaDetector v6 → Detects WHERE animals are (bounding boxes)")
print("2. Classification Model → Identifies WHAT SPECIES")
print("=" * 60)


# # ===================================================================
# # VEGETATION MAPPING MODELS (SAR + HYPERSPECTRAL)
# # ===================================================================

# print("\n" + "=" * 60)
# print("DOWNLOADING VEGETATION MAPPING MODELS")
# print("=" * 60)

# # 1. Prithvi EO 2.0 - Geospatial Foundation Model
# print("\n📥 Downloading Prithvi EO 2.0 (300M parameters)...")
# try:
#     from transformers import AutoModel
    
#     prithvi_model = AutoModel.from_pretrained(
#         "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
#         trust_remote_code=True,
#         cache_dir="~/.cache/huggingface/"
#     )
#     prithvi_model = prithvi_model.cuda() if torch.cuda.is_available() else prithvi_model
#     prithvi_model.eval()
    
#     print("✓ Prithvi EO 2.0 loaded successfully")
#     print(f"  - Architecture: Vision Transformer (ViT)")
#     print(f"  - Parameters: 300M")
#     print(f"  - Input: 6-band Sentinel-2 time-series")
#     print(f"  - Best for: Temporal vegetation classification")
# except Exception as e:
#     print(f"⚠️  Error loading Prithvi EO 2.0: {e}")
#     print("   Install dependencies: pip install transformers")

# # 2. BigEarthNet ResNet50 - SAR + Optical Fusion
# print("\n📥 Downloading BigEarthNet ResNet50...")
# try:
#     import timm
    
#     # Load ResNet50 pretrained on BigEarthNet
#     bigearth_model = timm.create_model(
#         'resnet50',
#         pretrained=False,  # We'll load custom weights
#         num_classes=19  # BigEarthNet v2 has 19 classes
#     )
    
#     # Download weights from Hugging Face
#     from huggingface_hub import hf_hub_download
    
#     weights_path = hf_hub_download(
#         repo_id="BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0",
#         filename="pytorch_model.bin",
#         cache_dir="~/.cache/huggingface/"
#     )
    
#     # Load weights
#     state_dict = torch.load(weights_path, map_location=device)
#     bigearth_model.load_state_dict(state_dict, strict=False)
#     bigearth_model = bigearth_model.cuda() if torch.cuda.is_available() else bigearth_model
#     bigearth_model.eval()
    
#     print("✓ BigEarthNet ResNet50 loaded successfully")
#     print(f"  - Architecture: ResNet50 (CNN)")
#     print(f"  - Parameters: 25M")
#     print(f"  - Input: 12 bands (Sentinel-1 VH/VV + Sentinel-2)")
#     print(f"  - Best for: SAR+Optical fusion, fast inference")
# except Exception as e:
#     print(f"⚠️  Error loading BigEarthNet ResNet50: {e}")
#     print("   Install dependencies: pip install timm huggingface_hub")