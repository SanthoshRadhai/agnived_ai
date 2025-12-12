
import glob
from flask import Flask, request, jsonify
from pathlib import Path
import sys
import traceback
import os
import json
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Import the landcover pipeline
from landcover.LandCover import AOIConfig as LandcoverAOIConfig, DownloadConfig, run_landcover_pipeline
# Import the vegetation pipeline
from vegetation.Vegetation_Classification_pipeline import AOIConfig as VegAOIConfig, run_bigearth_rdnet

# GoogleVR4 imports
import GoogleVR4.core as core
from GoogleVR4.ObjectIdentifier import run_object_detection

app = Flask(__name__)

# Plant model for classification
PLANT_MODEL_ID = "juppy44/plant-identification-2m-vit-b"
plant_processor = AutoImageProcessor.from_pretrained(PLANT_MODEL_ID)
plant_model = AutoModelForImageClassification.from_pretrained(PLANT_MODEL_ID)

@app.route('/panos', methods=['POST'])
def panos_api():
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    count = data.get('count', 3)
    area_of_interest = data.get('area_of_interest', 100.0)
    min_distance = data.get('min_distance', 10.0)
    # Call the panoramic search/slice function
    result = core.find_panos_and_views(
        lat=lat,
        lon=lon,
        n=count,
        radius_m=area_of_interest,
        min_distance_m=min_distance
    )
    # Save result to out/panos_result.json
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'panos_result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    return jsonify(result)

@app.route('/detect_objects', methods=['POST'])
def detect_objects_api():
    data = request.get_json()
    image_path = data.get('image_path')
    labels = data.get('labels', ['tree', 'bushes'])
    # Call the object detection function
    result = run_object_detection(image_path, labels)
    return jsonify(result)

@app.route('/panos_detect_objects', methods=['POST'])
def panos_detect_objects_api():
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    count = data.get('count', 3)
    area_of_interest = data.get('area_of_interest', 100.0)
    min_distance = data.get('min_distance', 10.0)
    labels = data.get('labels', ['tree', 'bushes'])

    # Step 1: Run panos search
    pano_result = core.find_panos_and_views(
        lat=lat,
        lon=lon,
        n=count,
        radius_m=area_of_interest,
        min_distance_m=min_distance
    )
    # Save pano result JSON
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'panos_detect_objects_result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(pano_result, f, indent=2)

    # Step 2: For each pano image, run object detection
    detected_objects = []
    for img in pano_result.get('images', []):
        if img.get('pano_downloaded'):
            pano_path = img.get('pano_path')
            if pano_path and os.path.exists(pano_path):
                detect_result = run_object_detection(pano_path, labels)
                detected_objects.append({
                    'pano_id': img.get('id'),
                    'pano_path': pano_path,
                    'object_detection': detect_result
                })
            else:
                detected_objects.append({
                    'pano_id': img.get('id'),
                    'pano_path': pano_path,
                    'object_detection': 'Pano image not found.'
                })
        else:
            detected_objects.append({
                'pano_id': img.get('id'),
                'pano_path': img.get('pano_path'),
                'object_detection': 'Pano not downloaded.'
            })

    # Step 3: Return combined result
    combined_result = {
        'pano_result': pano_result,
        'detected_objects': detected_objects
    }
    return jsonify(combined_result)

@app.route('/classify_plant', methods=['POST'])
def classify_plant_api():
    data = request.get_json()
    image_path = data.get('image_path')
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "image_path not provided or file does not exist"}), 400
    try:
        image = Image.open(image_path)
        inputs = plant_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = plant_model(**inputs).logits
        pred = logits.softmax(dim=-1)[0]
        topk = torch.topk(pred, k=5)
        results = []
        for prob, idx in zip(topk.values, topk.indices):
            label = plant_model.config.id2label[idx.item()]
            results.append({"label": label, "probability": float(prob.item())})
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/panos_detect_and_classify', methods=['POST'])
def panos_detect_and_classify_api():
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    count = data.get('count', 3)
    area_of_interest = data.get('area_of_interest', 100.0)
    min_distance = data.get('min_distance', 10.0)
    labels = data.get('labels', ['tree', 'bushes'])

    # Step 1: Run panos search and object detection
    pano_result = core.find_panos_and_views(
        lat=lat,
        lon=lon,
        n=count,
        radius_m=area_of_interest,
        min_distance_m=min_distance
    )
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'panos_detect_and_classify_result.json')

    detected_objects = []
    classify_results = []

    for img in pano_result.get('images', []):
        if img.get('pano_downloaded'):
            pano_path = img.get('pano_path')
            if pano_path and os.path.exists(pano_path):
                detect_result = run_object_detection(pano_path, labels)
                detected_objects.append({
                    'pano_id': img.get('id'),
                    'pano_path': pano_path,
                    'object_detection': detect_result
                })

                # Step 2: For each detected crop, classify plant
                crop_dir = os.path.join(os.path.dirname(pano_path), '..', 'detected_crops')
                crop_dir = os.path.abspath(crop_dir)
                crop_images = glob.glob(os.path.join(crop_dir, '*.jpg'))
                crop_classifications = []
                for crop_img in crop_images:
                    # Internal call to classify_plant
                    try:
                        image = Image.open(crop_img)
                        inputs = plant_processor(images=image, return_tensors="pt")
                        with torch.no_grad():
                            logits = plant_model(**inputs).logits
                        pred = logits.softmax(dim=-1)[0]
                        topk = torch.topk(pred, k=5)
                        results = []
                        for prob, idx in zip(topk.values, topk.indices):
                            label = plant_model.config.id2label[idx.item()]
                            results.append({"label": label, "probability": float(prob.item())})
                        crop_classifications.append({
                            "crop_image": crop_img,
                            "classification": results
                        })
                    except Exception as e:
                        crop_classifications.append({
                            "crop_image": crop_img,
                            "error": str(e)
                        })
                classify_results.append({
                    'pano_id': img.get('id'),
                    'crop_classifications': crop_classifications
                })
            else:
                detected_objects.append({
                    'pano_id': img.get('id'),
                    'pano_path': pano_path,
                    'object_detection': 'Pano image not found.'
                })
        else:
            detected_objects.append({
                'pano_id': img.get('id'),
                'pano_path': img.get('pano_path'),
                'object_detection': 'Pano not downloaded.'
            })

    combined_result = {
        'pano_result': pano_result,
        'detected_objects': detected_objects,
        'classify_results': classify_results
    }
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(combined_result, f, indent=2)
    return jsonify(combined_result)

@app.route('/run_landcover', methods=['POST'])
def run_landcover():
    try:
        data = request.get_json()
        lon = float(data['lon'])
        lat = float(data['lat'])
        buffer_km = float(data.get('buffer_km', 3.0))
        date_start = data.get('date_start', '2024-10-01')
        date_end = data.get('date_end', '2024-11-15')
        scale = int(data.get('scale', 10))
        cloud_cover_max = int(data.get('cloud_cover_max', 20))
        parent_dir = Path(__file__).resolve().parent
        results_dir = parent_dir / "LandcoverResults"
        aoi_cfg = LandcoverAOIConfig(lon=lon, lat=lat, buffer_km=buffer_km)
        dl_cfg = DownloadConfig(
            output_dir=results_dir,
            date_start=date_start,
            date_end=date_end,
            scale=scale,
            cloud_cover_max=cloud_cover_max,
        )
        outputs = run_landcover_pipeline(aoi_cfg, dl_cfg)
        return jsonify({k: str(v) for k, v in outputs.items()})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/run_vegetation', methods=['POST'])
def run_vegetation():
    try:
        data = request.get_json()
        lon = float(data['lon'])
        lat = float(data['lat'])
        buffer_km = float(data.get('buffer_km', 3.0))
        mask_path = data.get('mask_path')
        if not mask_path:
            # Default to LandcoverResults/vegetation_mask.tif in parent dir
            parent_dir = Path(__file__).resolve().parent.parent
            mask_path = parent_dir / "LandcoverResults" / "vegetation_mask.tif"
        aoi_cfg = VegAOIConfig(lon=lon, lat=lat, buffer_km=buffer_km)
        res = run_bigearth_rdnet(aoi_cfg, veg_mask_path=Path(mask_path))
        result = {
            'aoi': vars(res.aoi) if hasattr(res.aoi, '__dict__') else str(res.aoi),
            'cube_path': str(res.cube_path),
            'viz_path': str(res.viz_path),
            'class_distribution': res.class_distribution,
            'tile_counts': res.tile_counts,
            'avg_confidence': res.avg_confidence,
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/run_landcover_and_vegetation', methods=['POST'])
def run_landcover_and_vegetation():
    try:
        data = request.get_json()
        lon = float(data['lon'])
        lat = float(data['lat'])
        buffer_km = float(data.get('buffer_km', 3.0))
        date_start = data.get('date_start', '2024-10-01')
        date_end = data.get('date_end', '2024-11-15')
        scale = int(data.get('scale', 10))
        cloud_cover_max = int(data.get('cloud_cover_max', 20))
        parent_dir = Path(__file__).resolve().parent
        results_dir = parent_dir / "LandcoverResults"
        # Landcover pipeline
        aoi_cfg = LandcoverAOIConfig(lon=lon, lat=lat, buffer_km=buffer_km)
        dl_cfg = DownloadConfig(
            output_dir=results_dir,
            date_start=date_start,
            date_end=date_end,
            scale=scale,
            cloud_cover_max=cloud_cover_max,
        )
        landcover_outputs = run_landcover_pipeline(aoi_cfg, dl_cfg)
        # Vegetation pipeline (use mask from landcover output)
        mask_path = landcover_outputs.get('vegetation_mask')
        veg_aoi_cfg = VegAOIConfig(lon=lon, lat=lat, buffer_km=buffer_km)
        veg_res = run_bigearth_rdnet(veg_aoi_cfg, veg_mask_path=Path(mask_path))
        veg_result = {
            'aoi': vars(veg_res.aoi) if hasattr(veg_res.aoi, '__dict__') else str(veg_res.aoi),
            'cube_path': str(veg_res.cube_path),
            'viz_path': str(veg_res.viz_path),
            'class_distribution': veg_res.class_distribution,
            'tile_counts': veg_res.tile_counts,
            'avg_confidence': veg_res.avg_confidence,
        }
        return jsonify({
            'landcover': {k: str(v) for k, v in landcover_outputs.items()},
            'vegetation': veg_result
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
