from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
import torch
from PIL import Image, ImageDraw, ImageFont
import os

TEST_IMAGE_PATH = "/home/ray/agnived_models/amazon 4.png"

# Model-specific image size requirements
MODEL_IMAGE_SIZES = {
    "AI4GSnapshotSerengeti": 224,  # Standard ImageNet size
    "Deepfaune": 224,              # Standard ImageNet size
    "AI4GAmazonRainforest": 224    # Standard ImageNet size
}

def show_image(img, title="Image"):
    """Display PIL image and wait for user to close"""
    img.show(title=title)
    input(f"\n[{title}] Press Enter to continue...")

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def expand_bbox(bbox, img_width, img_height, padding_percent=0.15):
    """Expand bounding box by percentage while staying within image bounds"""
    x1, y1, x2, y2 = bbox
    
    width = x2 - x1
    height = y2 - y1
    
    pad_w = width * padding_percent
    pad_h = height * padding_percent
    
    x1_new = max(0, x1 - pad_w)
    y1_new = max(0, y1 - pad_h)
    x2_new = min(img_width, x2 + pad_w)
    y2_new = min(img_height, y2 + pad_h)
    
    return [x1_new, y1_new, x2_new, y2_new]

def resize_for_classification(img, target_size=224):
    """Resize image maintaining aspect ratio with padding"""
    # Calculate scaling to fit within target_size
    width, height = img.size
    scale = min(target_size / width, target_size / height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize maintaining aspect ratio
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create padded image (center the resized image)
    padded_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    padded_img.paste(img_resized, (paste_x, paste_y))
    
    return padded_img

def load_classifier(model_name, device):
    """Load classifier with model-specific initialization"""
    if model_name == "Deepfaune":
        return pw_classification.DFNE(device=device)
    elif model_name == "AI4GSnapshotSerengeti":
        return pw_classification.AI4GSnapshotSerengeti(device=device, pretrained=True)
    elif model_name == "AI4GAmazonRainforest":
        return pw_classification.AI4GAmazonRainforest(device=device, pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def test_detection(device):
    """Test MegaDetector v6"""
    print_section("STEP 1: ANIMAL DETECTION")
    
    print("Loading MegaDetector v6...")
    detector = pw_detection.MegaDetectorV6(
        device=device,
        pretrained=True,
        version="MDV6-yolov10-e"
    )
    print("✓ Detector loaded\n")
    
    img = Image.open(TEST_IMAGE_PATH)
    print(f"Original image: {img.size[0]}x{img.size[1]} pixels")
    show_image(img, "Original Image")
    
    print("\nRunning detection...")
    results = detector.single_image_detection(TEST_IMAGE_PATH)
    
    detections_obj = results['detections'][0]
    category_map = {0: "Animal", 1: "Person", 2: "animal"}
    animal_detections = []
    
    print(f"Total detections: {len(detections_obj.xyxy)}\n")
    
    for idx in range(len(detections_obj.xyxy)):
        bbox = detections_obj.xyxy[idx]
        confidence = float(detections_obj.confidence[idx])
        category_id = int(detections_obj.class_id[idx])
        
        category = category_map.get(category_id, "Unknown")
        print(f"Detection #{idx + 1}: {category} ({confidence:.1%})")
        
        if category_id == 0:
            animal_detections.append({
                'bbox': bbox.tolist(),
                'confidence': confidence
            })
    
    if len(animal_detections) == 0:
        print("\n❌ No animals detected!")
        return None
    
    # Visualize detection
    img_viz = img.copy()
    draw = ImageDraw.Draw(img_viz)
    
    first_animal = animal_detections[0]
    bbox = first_animal['bbox']
    conf = first_animal['confidence']
    
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    draw.text((x1, y1 - 20), f"Animal: {conf:.1%}", fill="red")
    
    print(f"\n✓ Using Detection #1 for classification")
    print(f"  Confidence: {conf:.1%}")
    print(f"  BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    
    show_image(img_viz, "Detection Result")
    
    return {
        'image': img,
        'detection': first_animal
    }

def test_all_classifiers(device, detection_data):
    """Test all classification models"""
    print_section("STEP 2: SPECIES CLASSIFICATION - ALL MODELS")
    
    CLASSIFICATION_MODELS = {
        "AI4GSnapshotSerengeti": "Snapshot Serengeti - 48 African species",
        "Deepfaune": "European wildlife - 20+ mammals/birds",
        "AI4GAmazonRainforest": "Amazon species (tropical/rainforest)"
    }
    
    img = detection_data['image']
    bbox_original = detection_data['detection']['bbox']
    
    # Expand bbox for better context
    bbox_expanded = expand_bbox(
        bbox_original,
        img.size[0],
        img.size[1],
        padding_percent=0.15
    )
    
    x1, y1, x2, y2 = bbox_expanded
    print(f"Original BBox: [{bbox_original[0]:.0f}, {bbox_original[1]:.0f}, {bbox_original[2]:.0f}, {bbox_original[3]:.0f}]")
    print(f"Expanded BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] (+15% padding)\n")
    
    # Crop animal
    cropped = img.crop((x1, y1, x2, y2))
    print(f"Cropped animal (original): {cropped.size[0]}x{cropped.size[1]} pixels")
    
    # Test each classifier
    all_results = []
    
    for model_name, description in CLASSIFICATION_MODELS.items():
        print(f"\n{'─' * 60}")
        print(f"Testing: {model_name}")
        print(f"Purpose: {description}")
        print(f"{'─' * 60}")
        
        try:
            # Resize for this specific model
            target_size = MODEL_IMAGE_SIZES.get(model_name, 224)
            cropped_resized = resize_for_classification(cropped, target_size)
            print(f"Resized for {model_name}: {cropped_resized.size[0]}x{cropped_resized.size[1]} pixels")
            
            # Save resized crop
            crop_path = f"/tmp/animal_crop_{model_name}.jpg"
            cropped_resized.save(crop_path)
            
            # Load classifier
            print(f"Loading {model_name}...")
            classifier = load_classifier(model_name, device)
            print("✓ Loaded")
            
            # Run classification
            print("Running classification...")
            result = classifier.single_image_classification(crop_path)
            
            if isinstance(result, dict) and 'prediction' in result:
                species = result['prediction']
                species_conf = result.get('confidence', 0.0)
                
                print(f"🐾 Species: {species}")
                print(f"   Confidence: {species_conf:.1%}")
                
                # Routing decision
                if species_conf >= 0.75:
                    routing = "✅ Auto-approved"
                elif species_conf >= 0.50:
                    routing = "⚠️  Medium-priority review"
                else:
                    routing = "🚨 High-priority review"
                
                print(f"   Routing: {routing}")
                
                all_results.append({
                    'model': model_name,
                    'species': species,
                    'confidence': species_conf,
                    'routing': routing
                })
                
                # Show annotated crop (original size for visibility)
                crop_annotated = cropped.copy()
                draw = ImageDraw.Draw(crop_annotated)
                text = f"{model_name}\n{species}\n{species_conf:.1%}"
                draw.text((10, 10), text, fill="green")
                show_image(crop_annotated, f"{model_name}: {species}")
                
            else:
                print(f"⚠️  Unexpected result: {result}")
                all_results.append({
                    'model': model_name,
                    'species': 'ERROR',
                    'confidence': 0.0,
                    'routing': '❌ Failed'
                })
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'model': model_name,
                'species': 'ERROR',
                'confidence': 0.0,
                'routing': '❌ Failed'
            })
    
    return all_results

def main():
    print("=" * 60)
    print("AGNIVED FAUNA DETECTION PIPELINE TEST")
    print("MULTI-MODEL COMPARISON")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\n❌ Image not found: {TEST_IMAGE_PATH}")
        return
    
    # Step 1: Detection
    detection_data = test_detection(device)
    if detection_data is None:
        return
    
    # Step 2: Classification with all models
    classification_results = test_all_classifiers(device, detection_data)
    
    # Step 3: Comparison Summary
    print_section("CLASSIFICATION COMPARISON SUMMARY")
    
    print(f"\nDetection Confidence: {detection_data['detection']['confidence']:.1%}\n")
    
    print(f"{'Model':<30} {'Species':<25} {'Confidence':<12} {'Routing'}")
    print("─" * 90)
    
    for result in classification_results:
        print(f"{result['model']:<30} {result['species']:<25} {result['confidence']:>6.1%}      {result['routing']}")
    
    # Best prediction
    print("\n" + "─" * 90)
    valid_results = [r for r in classification_results if r['species'] != 'ERROR']
    
    if valid_results:
        best = max(valid_results, key=lambda x: x['confidence'])
        print(f"\n🏆 HIGHEST CONFIDENCE: {best['model']}")
        print(f"   Species: {best['species']}")
        print(f"   Confidence: {best['confidence']:.1%}")
        print(f"   {best['routing']}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    if valid_results:
        # Check if there's consensus
        species_list = [r['species'] for r in valid_results]
        if len(set(species_list)) == 1:
            print("✅ All models agree on species identification")
        else:
            print("⚠️  Models disagree - manual review recommended")
            print(f"   Predictions: {', '.join(set(species_list))}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()