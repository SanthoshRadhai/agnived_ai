# GoogleVR4 API Server

This directory contains a Flask-based API server for panoramic image search, object detection, and plant classification using Mapillary imagery and deep learning models.

## API Endpoints

### 1. `/panos` (POST)
**Description:**
Finds well-distributed Mapillary 360° images near a given latitude/longitude, splits each panorama into 4 normal views, and saves images.

**Payload Example:**
```json
{
  "lat": 24.492786100000018,
  "lon": 77.34341670000003,
  "count": 3,
  "area_of_interest": 100,
  "min_distance": 20
}
```
**Result Storage:**
- Panoramas: `out/panos/<image_id>.jpg`
- Views: `out/views/<image_id>_front.jpg`, ...
- JSON summary: `out/panos_result.json`

---

### 2. `/detect_objects` (POST)
**Description:**
Runs object detection (Grounding DINO) on a given image and returns/crops detected objects.

**Payload Example:**
```json
{
  "image_path": "path/to/image.jpg",
  "labels": ["tree", "bushes"]
}
```
**Result Storage:**
- Cropped objects: `detected_crops/`
- JSON response: returned in API

---

### 3. `/classify_plant` (POST)
**Description:**
Classifies a single image using a plant identification model and returns top 5 predictions.

**Payload Example:**
```json
{
  "image_path": "path/to/image.jpg"
}
```
**Result Storage:**
- JSON response: returned in API

---

### 4. `/panos_detect_objects` (POST)
**Description:**
Runs panoramic search, saves images, and performs object detection on each panorama. Returns combined results.

**Payload Example:**
```json
{
  "lat": 24.492786100000018,
  "lon": 77.34341670000003,
  "count": 3,
  "area_of_interest": 100,
  "min_distance": 20,
  "labels": ["tree", "bushes"]
}
```
**Result Storage:**
- JSON summary: `out/panos_detect_objects_result.json`
- Images/crops: as above

---

### 5. `/panos_detect_and_classify` (POST)
**Description:**
Runs panoramic search, object detection, and plant classification for each detected crop. Returns and stores all results.

**Payload Example:**
```json
{
  "lat": 24.492786100000018,
  "lon": 77.34341670000003,
  "count": 3,
  "area_of_interest": 100,
  "min_distance": 20,
  "labels": ["tree", "bushes"]
}
```
**Result Storage:**
- JSON summary: `out/panos_detect_and_classify_result.json`
- Images/crops: as above
- Plant classification: included in JSON

---

## Output Directories
- `out/panos/` : Downloaded panoramic images
- `out/views/` : Perspective views from panos
- `out/` : JSON result files
- `detected_crops/` : Cropped object images from detection

## Usage
1. Start the Flask server: `python app.py`
2. Use the provided Python client scripts or `requests` to call the endpoints.
3. Check the `out/` and `detected_crops/` directories for results.

## Requirements
- Python 3.8+
- Install dependencies: `pip install -r requirements.txt`
- Set your Mapillary token in `.env` as `MAPILLARY_TOKEN=MLY|your_token_here`

## Notes
- All API responses are in JSON format.
- Large queries may take time due to image downloads and model inference.
- Ensure all required models and dependencies are installed for object detection and classification.
