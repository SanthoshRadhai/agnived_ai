"""
Test script for the fast pipeline:
/run_panos_and_plant_identification

This endpoint:
1. Finds panoramic images from Mapillary
2. Detects objects (trees, bushes, vegetation)
3. Identifies plant species for each detected crop
4. Skips landcover and vegetation analysis

Much faster than the full pipeline!
"""

import requests
import json
import time

# API endpoint
API_URL = "http://127.0.0.1:5000/run_panos_and_plant_identification"

# Test payload
payload = {
    # Panorama parameters (small AOI for street-level)
    "lat": 28.56931789780799,
    "lon": 77.312035309223,
    "panos_lat": 28.56931789780799,
    "panos_lon": 77.312035309223,
    "panos_count": 3,
    "panos_area_of_interest": 100.0,
    "panos_min_distance": 20.0,
    "panos_labels": ["tree", "bushes", "vegetation"],
    
    # Other parameters (not used in this endpoint, but required by request model)
    "buffer_km": 3.1,
    "date_start": "2024-10-01",
    "date_end": "2024-11-15",
    "scale": 10,
    "cloud_cover_max": 20
}

print("=" * 80)
print("Testing Fast Pipeline: Panos + Plant Identification (No Landcover/Vegetation)")
print("=" * 80)
print(f"\nRequest URL: {API_URL}")
print(f"\nPayload:")
print(json.dumps(payload, indent=2))

print("\n" + "=" * 80)
print("Sending request (this may take 2-5 minutes)...")
print("=" * 80)

try:
    start_time = time.time()
    response = requests.post(API_URL, json=payload, timeout=600)  # 10 minute timeout
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Request completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n" + "=" * 80)
        print("RESPONSE SUMMARY")
        print("=" * 80)
        
        # Summary stats
        if 'summary' in result:
            summary = result['summary']
            print("\n📊 SUMMARY:")
            print(f"  Total Panoramic Images: {summary.get('total_panos', 0)}")
            print(f"  Total Objects Detected: {summary.get('total_objects_detected', 0)}")
            print(f"  Total Plants Identified: {summary.get('total_plants_identified', 0)}")
        
        # Panorama results
        if 'panos' in result:
            panos = result['panos']
            pano_result = panos.get('pano_result', {})
            
            print("\n📸 PANORAMIC IMAGES:")
            print(f"  Found: {pano_result.get('found')}")
            print(f"  Count Returned: {pano_result.get('count_returned')}")
            print(f"  Area of Interest: {pano_result.get('area_of_interest_m')}m")
            print(f"  Min Distance: {pano_result.get('min_distance_m')}m")
            
            # Object detection results
            detected_objects = panos.get('detected_objects', [])
            print(f"\n🔍 OBJECT DETECTION:")
            print(f"  Total detections: {len(detected_objects)}")
            for i, obj in enumerate(detected_objects, 1):
                print(f"\n  Pano {i}:")
                print(f"    Pano ID: {obj.get('pano_id')}")
                print(f"    Pano Path: {obj.get('pano_path')}")
                if isinstance(obj.get('object_detection'), dict):
                    crops_count = len(obj['object_detection'].get('crops', []))
                    print(f"    Objects Detected: {crops_count}")
                else:
                    print(f"    Detection: {obj.get('object_detection')}")
            
            # Plant identification results
            plant_id_results = panos.get('plant_identification_results', [])
            print(f"\n🌱 PLANT SPECIES IDENTIFICATION:")
            print(f"  Total panos analyzed: {len(plant_id_results)}")
            
            for i, pano_plant_result in enumerate(plant_id_results, 1):
                print(f"\n  Pano {i} - ID: {pano_plant_result.get('pano_id')}")
                print(f"    Path: {pano_plant_result.get('pano_path')}")
                
                if 'error' in pano_plant_result:
                    print(f"    Error: {pano_plant_result['error']}")
                else:
                    crop_identifications = pano_plant_result.get('crop_plant_identifications', [])
                    identified_count = len([c for c in crop_identifications if 'predictions' in c])
                    print(f"    Crops identified: {identified_count}/{len(crop_identifications)}")
                    
                    for j, crop in enumerate(crop_identifications, 1):
                        print(f"\n      Crop {j}:")
                        print(f"        Object Type: {crop.get('object_label')} (score: {crop.get('object_score'):.2f})")
                        
                        if 'error' in crop:
                            print(f"        Error: {crop['error']}")
                        else:
                            predictions = crop.get('predictions', [])
                            print(f"        Top 5 Species:")
                            for k, pred in enumerate(predictions[:5], 1):
                                confidence = pred.get('confidence_percentage', '0%')
                                print(f"          {k}. {pred.get('species')} - {confidence}")
                            
                            top_pred = crop.get('top_prediction')
                            if top_pred:
                                print(f"        Identified Species: {top_pred.get('species')} ({top_pred.get('confidence_percentage')})")
        
        # Full response
        print("\n" + "=" * 80)
        print("FULL RESPONSE (JSON):")
        print("=" * 80)
        print(json.dumps(result, indent=2))
        
        print("\n" + "=" * 80)
        print("✓ Test PASSED")
        print("=" * 80)
        print("\nResult saved to: E:\\6thSem\\GLYTCH'25\\PipeLine3.0\\out\\panos_and_plant_identification_result.json")
        
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\n✗ Connection Error: Could not connect to the server.")
    print("Make sure the FastAPI server is running:")
    print("  conda activate glytch25")
    print("  cd E:\\6thSem\\GLYTCH'25\\PipeLine3.0")
    print("  python server_fastapi.py")
except requests.exceptions.Timeout:
    print("\n✗ Timeout Error: Request took too long.")
    print("This pipeline takes 2-5 minutes depending on image availability.")
except Exception as e:
    print(f"\n✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()
