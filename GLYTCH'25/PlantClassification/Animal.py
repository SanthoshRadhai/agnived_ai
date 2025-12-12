import pandas as pd
import open_clip
from PIL import Image
import torch
import numpy as np
import requests
from io import BytesIO
from pathlib import Path
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def download_sample_species_list():
    """Download sample species list since TreeOfLife-200M CSV isn't directly available"""
    # Sample wildlife species list (expandable to 1000s)
    sample_species = [
        # Indian wildlife
        "Panthera tigris tigris", "Elephas maximus indicus", "Panthera uncia", 
        "Pavo cristatus", "Rhinoceros unicornis", "Platanista gangetica",
        # African wildlife
        "Equus quagga", "Loxodonta africana", "Giraffa camelopardalis", 
        "Acinonyx jubatus", "Crocuta crocuta",
        # Global mammals/birds
        "Ursus arctos", "Canis lupus", "Falco peregrinus", "Bubo bubo", "mouse"
    ] * 100  # Repeat for batch testing
    
    # Save as CSV for consistency
    df = pd.DataFrame({'scientific_name': sample_species})
    df.to_csv('tree_of_life_species_sample.csv', index=False)
    print(f"✅ Created sample species list: {len(sample_species)} entries")
    return sample_species

def load_species_list(csv_path="tree_of_life_species_sample.csv", max_species=1000):
    """Load species list from CSV or create sample"""
    if Path(csv_path).exists():
        species_list = pd.read_csv(csv_path)['scientific_name'].tolist()
    else:
        print("📥 No species CSV found. Creating sample...")
        species_list = download_sample_species_list()
    
    # Limit for memory/performance
    species_list = species_list[:max_species]
    print(f"📊 Loaded {len(species_list)} species for classification")
    return species_list

def classify_any_species(model, preprocess, tokenizer, device, image_path, species_list, top_k=10, batch_size=100):
    """Classify image against large species list with batching"""
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Batch process species prompts for memory efficiency
    all_scores = []
    
    for i in tqdm(range(0, len(species_list), batch_size), desc="Classifying"):
        batch_species = species_list[i:i+batch_size]
        text_tokens = tokenizer([f"a photo of {s}" for s in batch_species]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            
            # Normalize and compute cosine similarity
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Collect scores
        batch_scores = similarities.cpu().numpy()[0]
        all_scores.extend(zip(batch_species, batch_scores))
    
    # Get top predictions
    all_scores.sort(key=lambda x: x[1], reverse=True)
    top_results = all_scores[:top_k]
    
    return top_results

def main():
    """Main execution function"""
    print("🚀 BioCLIP 2 - Zero-Shot Wildlife Species Classification")
    print("=" * 60)
    
    # 1. Load BioCLIP 2 model
    print("📥 Loading BioCLIP 2 model...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Model loaded on {device}")
    
    # 2. Load species list
    species_list = load_species_list(max_species=10000)  # Adjust as needed
    
    # 3. Classify image
    image_path = input("\n📸 Enter image path (or URL): ").strip()
    
    # Handle URL or local file
    if image_path.startswith(('http://', 'https://')):
        print("🌐 Downloading image...")
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        temp_path = "temp_downloaded_image.jpg"
        image.save(temp_path)
        image_path = temp_path
    elif not Path(image_path).exists():
        print("❌ Image not found. Using sample...")
        return
    
    print(f"🔍 Classifying: {Path(image_path).name}")
    
    # 4. Run classification
    start_time = time.time()
    results = classify_any_species(model, preprocess_val, tokenizer, device, 
                                  image_path, species_list, top_k=10, batch_size=100)
    end_time = time.time()
    
    # 5. Display results
    print("\n" + "="*60)
    print("🏆 TOP 10 SPECIES PREDICTIONS")
    print("="*60)
    for i, (species, prob) in enumerate(results, 1):
        print(f"{i:2d}. {species:<40} {prob:>6.2f}%")
    
    print(f"\n⏱️  Inference time: {end_time-start_time:.2f}s")
    print(f"🎯 Searched against {len(species_list):,} species")
    
    # Cleanup
    if 'temp_downloaded_image.jpg' in image_path:
        Path(image_path).unlink()

if __name__ == "__main__":
    main()
