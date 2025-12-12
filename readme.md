# AgniVed Geospatial & Wildlife Analysis Backend

This repository contains the Python fullstack environment with a specialised backend for the AgniVed project. It implements:

- A **Dynamic World + Sentinel-2** land-cover pipeline (AOI → composites → classification/masks/visualizations)
- A **BigEarthNet v2.0 (reBEN)** vegetation classification pipeline running rdnet on high-quality Sentinel-2 patches
- A **wildlife detection + species classification** pipeline using PytorchWildlife (MegaDetector v6 + three classifiers) on:
  - Local videos (developer test scripts)
  - YouTube livestreams/videos, via a live inference engine
- A **FastAPI backend** (`Python Backend/main_api.py`) orchestrating these pipelines and exposing HTTP endpoints

All code and experiments are designed to run locally in a Python 3.11 virtual environment.

---

## 1. Repository Layout

At the top level:

### Root Files

- **`.gitignore`**

- **`agniv_requirements.txt`**  
  Base requirements snapshot (Linux-oriented). A trimmed set is installed into the 3.11 venv.

- **`download_models.py`**  
  Helper script to pre-download PytorchWildlife detection/classification models and (optionally) vegetation models (Prithvi, BigEarthNet ResNet50). Mostly for model caching and testing.

### Directories

- **`agnived_env/`**  
  Python 3.11 virtual environment directory (created locally). Contains:
  - `pyvenv.cfg`, `Lib/`, `Scripts/`, etc.
  - This env is what the backend uses; not meant to be edited manually.

- **`Final_Res_DW/`**  
  Production output directory for the Dynamic World + Sentinel-2 land-cover pipeline:
  - `sentinel2_hyperspectral.tif` – 12-band S2 composite
  - `land_cover_classification.tif` – DW class labels (0–8)
  - `land_cover_probabilities.tif` – DW class probabilities
  - `vegetation_mask.tif` – combined vegetation probability mask (trees/grass/crops/shrub/flooded_vegetation)
  - `agnived_cover_analysis.png` – 6-panel land-cover visualization
  - `mask_*.png` – per-class binary masks
  - `metadata.json` – AOI + statistics

- **`Hyperspectral models/`**  
  Placeholder for advanced models:
  - `bigearth/` – BigEarthNet experiments (rdnet/convnext/etc)
  - `Prithvi model/` – Prithvi EO 2.0 experiments (commented in `download_models.py`)

- **`Python Backend/`**  
  Main backend code:
  - `main_api.py` – FastAPI app (described in detail below)
  - `reben/`, `reben_publication/` – local clone/extract of the reBEN BigEarthNet v2.0 model code (rdnet, convnext, etc.)
  - `S2 Landcover pipeline/` – Dynamic World + Sentinel-2 land-cover pipeline
  - `S2 Vegetation Classification pipeline/` – BigEarthNet rdnet S2 vegetation classification pipeline
  - `Video_inference_engine/` – YouTube video/live inference engine for wildlife detection & classification
  - `video_results/` – JSON and other outputs from video inference

- **`Test_CameraTraps/`**  
  Developer/test scripts for the wildlife pipeline:
  - `Test_models.py` – validates single-image detection and species classification models
  - `test_video.py` – reference implementation of local video real-time inference

- **`Test_results_DW/`**  
  Earlier test outputs from the land-cover pipeline (`metadata.json`, etc.)

- **`Test_Satellite/`**  
  Prototyping and research notebooks/scripts for:
  - Dynamic World + S2 downloader (`TestClassificationDownload.py`)
  - BigEarthNet S2 and S1+S2 experiments (`TestBigEarthrdnet.py`, `TestBigEarthS1S2.py`)
  - reBEN code (`reben_publication/`)
  - These are not used by the production backend directly, but they define the reference behaviour the backend implements.

---

## 2. Python Environment

### 2.1. Python / venv

- **Target Python version:** 3.11
- **Local venv:** `agnived_env` at the repo root

To recreate:

```bash
python3.11 -m venv agnived_env
source agnived_env/bin/activate  # On Windows: agnived_env\Scripts\activate
pip install -r agniv_requirements.txt  # (or subset)
```

Environment includes (core):

- `fastapi`, `uvicorn`
- `earthengine-api`, `geemap`, `rasterio`, `numpy`, `matplotlib`
- `PytorchWildlife`, `torch`, `torchvision`, `ultralytics`
- `yt-dlp`, `opencv-python`
- `lightning`, `timm`, `transformers` (for reBEN / future models)

**Note:** `agniv_requirements.txt` is Linux-centric and includes ROS, GDAL, etc. For Windows, installation is typically done with a subset (only what's needed for these pipelines).

---

## 3. Dynamic World + Sentinel-2 Land-Cover Pipeline

### 3.1. Location

**Implementation:**  
`Python Backend/S2 Landcover pipeline/Download_Classify.py`

**Key public pieces:**

- `s2_landcover.AOIConfig`
- `s2_landcover.DownloadConfig`
- `s2_landcover.run_landcover_pipeline(aoi_cfg, dl_cfg)` – End-to-end pipeline

### 3.2. Data Sources

**Sentinel-2:** `COPERNICUS/S2_SR_HARMONIZED`  
Bands used:
```python
["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
```

**Dynamic World:** `GOOGLE/DYNAMICWORLD/V1`  
Classes 0–8:
- 0: Water
- 1: Trees
- 2: Grass
- 3: Flooded Vegetation
- 4: Crops
- 5: Shrub & Scrub
- 6: Built
- 7: Bare
- 8: Snow & Ice

Probability bands:
```python
["water","trees","grass","flooded_vegetation","crops","shrub_and_scrub","built","bare","snow_and_ice"]
```

**AOI is defined as:**
- Center: `(lon, lat)`
- Radius: buffer in km, converted to meters
- Geometry: point → buffer → `bounds()` → rectangular AOI

### 3.3. Pipeline Steps

#### 1. Initialize Earth Engine

`init_ee` (called from API):
- Tries `ee.Initialize(project=ee_project)`
- On failure, calls `ee.Authenticate()` → then `ee.Initialize`

#### 2. AOI

`create_aoi`:
- Builds `ee.Geometry.Point([lon, lat])`
- Buffers by `buffer_km * 1000` m
- Uses `.bounds()` for a rectangular patch

#### 3. Sentinel-2 composite

**`load_sentinel2(aoi, cfg)`:**
- Filters by bounds, `date_start`–`date_end`, `CLOUDY_PIXEL_PERCENTAGE < cloud_cover_max`

**`create_sentinel2_composite(collection)`:**
- Median composite across all selected S2 images
- Selects the full 12-band stack

**`download_geotiff`:**
- Uses `image.getDownloadURL` with:
  - `scale = cfg.scale` (typically 10 m)
  - `region = aoi`
  - `filePerBand = False`, `format = GEO_TIFF`
- Streams to local `sentinel2_hyperspectral.tif` in `output_dir`

#### 4. Dynamic World classification & probabilities

**`load_dynamic_world(aoi, cfg)`:**
- Filter by AOI and same date window

**`create_classification_composite(dw_collection)`:**
- Mode composite of label band (0–8)

**`create_probability_composite(dw_collection)`:**
- Mean composite across the probability bands listed above

**Download to:**
- `land_cover_classification.tif` (`bands=['label']`)
- `land_cover_probabilities.tif` (all probability bands)

#### 5. Vegetation mask

**`create_vegetation_mask(dw_collection, threshold)`:**

Uses DW probability bands for vegetation-related classes:
- Trees
- Grass
- Crops
- Shrub and scrub
- Flooded vegetation

Process:
1. Computes mean probability across time
2. Reduces across those vegetation bands via `Reducer.max()`
3. Thresholds: `veg_mask = [p_max ≥ threshold]`
4. In production: threshold ≈ 0.27
5. Download as `vegetation_mask.tif`

#### 6. Statistics

**`calculate_statistics(classification_path, cfg)`:**
- Reads classification raster
- Computes class pixel counts using `np.unique`
- Converts to area (km²) using pixel area = `scale²`
- Computes percentage coverage
- Returns a dict keyed by class name with:
  - `pixels`, `area_km2`, `percentage`, `description`

#### 7. Visualization

**`create_visualizations`:**

Creates AgniVed cover analysis 6-panel PNG:
1. Sentinel-2 true color (RGB: B4–B3–B2)
2. DW classification map with legend
3. Land-cover percentage bar chart (horizontal, per class)
4. Sentinel-2 false color (NIR–R–G) with vegetation emphasized
5. Trees probability heatmap
6. Built area probability heatmap

Also:
- Calls `create_class_masks` to create per-class binary mask PNGs

#### 8. Metadata

**`save_metadata(stats, aoi, cfg)`:**
- Serializes AOI bounds, date range, scale, and land-cover statistics
- Writes JSON to `metadata.json`

#### 9. Wrapper

**`run_landcover_pipeline(aoi_cfg, dl_cfg)`:**
- Orchestrates all steps above
- Returns paths to:
  - `sentinel2`, `classification`, `probabilities`, `vegetation_mask`, `visualization`, `metadata`

---

## 4. BigEarthNet v2.0 Vegetation Classification (rdnet S2)

### 4.1. Location

**Backend implementation:**  
`Python Backend/S2 Vegetation Classification pipeline/Vegetation_Classification_pipeline.py`

**Base reference for behaviour:**  
`TestBigEarthrdnet.py` and `BigEarthNetv2_0_ImageClassifier.py`

### 4.2. Model

**reBEN BigEarthNet v2.0 S2-only model:** `rdnet_base-s2-v0.2.0`

**Band order:**
```python
["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]
# (aliases ["B2", "B3", ...])
```

**Classes (19):**
- "Agro-forestry areas"
- "Arable land"
- "Beaches, dunes, sands"
- "Broad-leaved forest"
- ...
- "Urban fabric"

(see `BEN_CLASSES` in the pipeline file)

**Normalization:**

DN-space band statistics `[BEN_MEAN, BEN_STD]` (copied from test script):

```python
tile_norm = (tile - BEN_MEAN) / BEN_STD
```

### 4.3. AOI & Configuration

**Dataclasses:**

**`AOIConfig`:**
```python
lon: float
lat: float
buffer_m: int  # buffer radius in meters
```

**`BigEarthConfig`:**
```python
date_start: str
date_end: str
cloud_cover_max: float
scale: int = 10  # meters per pixel
```

**`BigEarthResult`:**

Contains:
- `aoi`: AOIConfig
- `cube_path`: path to downloaded S2 stack
- `viz_path`: combined PNG (true color + rdnet class map + confidence)
- `class_distribution`: per-class tile percentage
- `tile_counts`: per-class tile counts
- `avg_confidence`: mean of top-1 probabilities
- `tiles_shape`: number of tiles in grid `(H_tiles, W_tiles)`

### 4.4. Pipeline

#### 1. Earth Engine

**`init_earth_engine`** mirrors `TestBigEarthrdnet`:
- `ee.Initialize(project="our-lamp-465108-a9")` or `ee.Authenticate()`

#### 2. AOI

**`build_aoi`:**
- Center point → buffer by `buffer_m` → `.bounds()`

#### 3. S2 composite

**`build_single_composite(aoi, cfg)`:**

Uses `COPERNICUS/S2_SR_HARMONIZED`:
- Filter by AOI, `date_start`–`date_end`, `CLOUDY_PIXEL_PERCENTAGE < cloud_cover_max`
- `select(S2_BANDS)`
- Median composite across time

This replicates the behaviour of `download_single_composite` in `TestBigEarthrdnet.py`.

#### 4. Download

**`download_composite`:**
- `image.getDownloadURL` → `bigearth_s2_stack.tif` in temporary dir

#### 5. Read Cube

**`read_cube`:**
- `rasterio.read()` → `(10, H, W)` array

#### 6. Tiling

**`tile_cube`:**
- `PATCH_SIZE = 120`
- Tiles the cube into non-overlapping 120×120 patches
- Requires `H // 120 > 0` and `W // 120 > 0` → approx 1.2 km × 1.2 km at 10 m

#### 7. Normalization

**`normalize_tiles`:**
- Applies per-band `BEN_MEAN` and `BEN_STD` from reBEN

#### 8. Model loading

**`load_reben_model`:**

Uses local `BigEarthNetv2_0_ImageClassifier.py`.

Calls:
```python
BigEarthNetv2_0_ImageClassifier(
    model_name="rdnet_base",
    bands="s2",
    ckpt_path="...",
    device=device
)
```

#### 9. Inference

- Converts tile batch to PyTorch tensor, moves to GPU if available
- `logits = model(tensor)` → `probs = sigmoid(logits)`
- Top-1 class per tile and its probability become:
  - `class_map` (2D grid)
  - `conf_map` (2D grid)

**Distribution & summary:**
- `np.bincount` over `class_map` → tile counts per BigEarthNet class
- Percentages per class

#### 10. Visualization

**True color:** uses S2 bands B4 (R), B3 (G), B2 (B), normalized to [0,1]

Produces a 3-panel figure:
1. True color S2
2. Top-1 class map (tab20 colormap with legend of unique classes)
3. Confidence heatmap (0–1)

Saves to `bigearth_rdnet_s2_results.png` under a `Results/` folder in the S2 Vegetation pipeline.

#### 11. Wrapper

**`run_bigearth_rdnet(aoi_cfg, be_cfg=None, device=None, out_dir=None, veg_mask_path=None)`**

(internally extended in `main_api.py` to optionally intersect with a `vegetation_mask.tif` if required)

---

## 5. Wildlife Detection + Classification (Video / YouTube)

### 5.1. PytorchWildlife models

**Code references:**
- `Test_models.py`
- `test_video.py`
- `download_models.py`

**Backend engine:**  
`Python Backend/Video_inference_engine/Video_inference.py`

**Models:**

**Detection:**
- **MegaDetector v6** (YOLOv10-based):
  ```python
  pw_detection.MegaDetectorV6(device=..., pretrained=True, version="MDV6-yolov10-e")
  ```
  Detects: Animal, Person, Vehicle

**Classification:**
- **AI4GSnapshotSerengeti** – 48 African species (camera-trap-trained)
- **DFNE / Deepfaune** – European wildlife
- **AI4GAmazonRainforest** – Amazon rainforest fauna

### 5.2. Reference: Local video pipeline (test_video.py)

`test_video.py` defines `VideoWildlifeDetector` with:

**`load_models()`:**
- Loads MegaDetector + 3 classifiers

**`detect_frame`:**
- Runs detection on each `FRAME_SKIP`'th frame:
  - Writes frame to `/tmp/frame_<idx>.jpg`
  - `MegaDetectorV6.single_image_detection` on that file
  - Filters for category 0 (Animal)

**`match_detection_to_track`:**
- Associates detections with tracks based on IoU

**`classify_detection_realtime`:**
- Crops/expands the animal bounding box
- Resizes (with aspect ratio + padding) to 224×224
- Saves to `/tmp/classify_crop.jpg`
- Calls each classifier's `single_image_classification`

**Frame buffer per track:**
- Stores some frames/detections
- Once enough are collected (`CLASSIFICATION_FRAMES_PER_TRACK`), classification is performed using the best detection frame

**`draw_realtime_annotations`:**
- Draws bounding boxes with species labels and confidences, plus per-model info

**`process_video_realtime`:**
- Real-time loop over a local file:
  - `cv2.VideoCapture`, fps, total_frames
  - Shows annotated frames with `cv2.imshow`
  - Pause/resume (p), quit (q)
- At the end, writes JSON summary via `save_results`

### 5.3. Backend: YouTube live inference

**Implementation:**  
`Python Backend/Video_inference_engine/Video_inference.py`

#### Configuration

```python
FRAME_SKIP = 2
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.3
TRACK_TIMEOUT = 30
CLASSIFICATION_FRAMES_PER_TRACK = 3
BUFFER_MINUTES = 10
```

#### Detector class

**`VideoWildlifeDetector`** mirrors the logic of the test script:

**`load_models()`**
- Loads MegaDetector v6
- Loads Serengeti / Deepfaune / Amazon classifiers

**`iou(box1, box2)`**

**`detect_animals(frame_rgb, frame_idx)`**
- Writes frame to `/tmp/frame_{frame_idx}.jpg`
- Filters to `category_id == 0` (Animal) detections

**`match_detection_to_track(detection, frame_idx)`**
- IoU-based track association with `TRACK_TIMEOUT` frames memory

**`classify_detection(frame_rgb, bbox)`**
- Expands bounding box (~15% padding)
- Crops and resizes to 224×224
- Saves to `/tmp/classify_crop.jpg`
- Calls each classifier's `single_image_classification`
- Picks highest-confidence species among models

**`draw_annotations(frame_rgb, frame_idx, total_frames=None)`**
- Draws bounding boxes and species labels, coloured by confidence
- Shows predictions from all models

**`process_stream_frame(frame_rgb, frame_idx, total_frames=None)`**

Called for each new frame by the YouTube loop.

Every `FRAME_SKIP` frames:
- Calls `detect_animals`
- Optionally restricts to single highest-confidence detection per frame to "focus on one animal"
- Feeds detections into tracking and classification (after enough frames per track)

On intermediate frames:
- Reuses last known bounding boxes for recent tracks

Returns annotated RGB frame.

**`save_results(video_name, fps)`**
- Writes summary JSON to `video_results/<video_name>_results.json`
- Includes track IDs, species, confidences, and time ranges

#### YouTube streaming

**`get_youtube_stream_url(youtube_url: str) -> str`**
- Uses `yt_dlp` to resolve the best MP4 stream URL (≤720p)

**`run_youtube_live_inference(youtube_url: str)`**

Process:
1. Prints device info and YouTube link
2. Resolves stream URL via `get_youtube_stream_url`
3. Opens `cv2.VideoCapture` on the stream URL
4. Estimates FPS ≈ 30
5. Keeps a deque buffer of `(frame_idx, frame)` covering up to `BUFFER_MINUTES * 60 * fps` frames (≈ last 10 minutes)

In each loop:
- Converts BGR frame → RGB
- Calls `detector.process_stream_frame(frame_rgb, frame_idx)`
- Converts back to BGR, shows via OpenCV window

**Future enhancement:** second pass over the buffered frames, akin to processing a small 10-minute clip with better context.

**Current behaviour:** near-real-time inference on incoming frames with a rolling buffer.

---

## 6. FastAPI Backend (main_api.py)

**Location:**  
`Python Backend/main_api.py`

This file exposes all major functionality as REST endpoints.

### 6.1. Module loading

To avoid import-path issues and keep test code separate, the backend dynamically loads the land-cover and vegetation modules via:

**`load_module_from_path`**

It binds:
- `s2_landcover.AOIConfig`, `DownloadConfig`, `run_landcover_pipeline`
- `s2_vegetation.AOIConfig`, `run_bigearth_rdnet` (lazy-loaded)

### 6.2. Request models

Pydantic models:

**`LandcoverRequest`:**
```python
lon: float
lat: float
buffer_km: float = 0.6  # defaults 0.6 → 600 m AOI radius
date_start: str
date_end: str
scale: int
cloud_cover_max: float
```

**`VegetationRequest`:**
```python
lon: float
lat: float
buffer_m: int = 600
use_mask: bool  # whether to intersect with vegetation_mask.tif from land-cover step
```

**`VideoRequest`:**
```python
youtube_url: str
```

**`PipelineRequest`:**

Combined config for both land-cover and vegetation:
```python
lon: float
lat: float
buffer_km: float  # for DW
buffer_m: int     # for BigEarth
date_start: str
date_end: str
scale: int
cloud_cover_max: float
```

### 6.3. Endpoints

#### 6.3.1. `/landcover/dw` (POST)

Runs the Dynamic World + Sentinel-2 land-cover pipeline.

**Input:** `LandcoverRequest`

**Implementation:**

```python
@app.post("/landcover/dw")
async def run_landcover_analysis(req: LandcoverRequest):
    # Build AOI and DownloadConfig
    aoi_cfg = DW_AOIConfig(lon=req.lon, lat=req.lat, buffer_km=req.buffer_km)
    dl_cfg = DownloadConfig(
        date_start=req.date_start,
        date_end=req.date_end,
        scale=req.scale,
        cloud_cover_max=req.cloud_cover_max,
        output_dir="Final_Res_DW"
    )
    
    # Run pipeline
    results = run_landcover_pipeline(aoi_cfg, dl_cfg)
    
    # Return paths
    return {
        "status": "success",
        "results": results
    }
```

**Returns:**
```json
{
  "status": "success",
  "results": {
    "sentinel2": "path/to/sentinel2_hyperspectral.tif",
    "classification": "path/to/land_cover_classification.tif",
    "probabilities": "path/to/land_cover_probabilities.tif",
    "vegetation_mask": "path/to/vegetation_mask.tif",
    "visualization": "path/to/agnived_cover_analysis.png",
    "metadata": "path/to/metadata.json"
  }
}
```

#### 6.3.2. `/vegetation/bigearth` (POST)

Runs the BigEarthNet rdnet S2 vegetation classifier.

**Input:** `VegetationRequest`

**Optionally uses vegetation mask from land-cover step:**

```python
veg_mask_path = None
if req.use_mask:
    veg_mask_path = "Final_Res_DW/vegetation_mask.tif"
```

**Implementation:**

```python
@app.post("/vegetation/bigearth")
async def run_vegetation_classification(req: VegetationRequest):
    # Lazy-load vegetation module
    if s2_vegetation is None:
        load_vegetation_module()
    
    # Build AOI
    aoi_cfg = VEG_AOIConfig(lon=req.lon, lat=req.lat, buffer_m=req.buffer_m)
    
    # Determine mask path
    veg_mask_path = "Final_Res_DW/vegetation_mask.tif" if req.use_mask else None
    
    # Run BigEarth pipeline
    result = s2_vegetation.run_bigearth_rdnet(
        aoi_cfg,
        veg_mask_path=veg_mask_path
    )
    
    return {
        "status": "success",
        "result": result
    }
```

**Returns:**
```json
{
  "status": "success",
  "result": {
    "aoi": {...},
    "cube_path": "path/to/bigearth_s2_stack.tif",
    "viz_path": "path/to/bigearth_rdnet_s2_results.png",
    "class_distribution": {...},
    "tile_counts": {...},
    "avg_confidence": 0.85,
    "tiles_shape": [10, 10]
  }
}
```

**Note:** This endpoint expects that `/landcover/dw` has already been run if `use_mask` is `True`.

#### 6.3.3. `/video/classify` (POST)

Starts a background YouTube live inference session.

**Input:** `VideoRequest` with `youtube_url`

**Implementation:**

```python
@app.post("/video/classify")
async def classify_video(req: VideoRequest):
    # Start inference in background thread
    thread = threading.Thread(
        target=run_youtube_live_inference,
        args=(req.youtube_url,),
        daemon=True
    )
    thread.start()
    
    return {
        "status": "started",
        "message": "YouTube inference running in background",
        "url": req.youtube_url
    }
```

**Returns immediately:**
```json
{
  "status": "started",
  "message": "YouTube inference running in background",
  "url": "https://youtube.com/watch?v=..."
}
```

A local OpenCV window will appear on the machine where the backend is running, showing the annotated video. This is intended for local prototyping.

#### 6.3.4. `/files/image` (GET)

Serves any generated image/GeoTIFF under the project root.

**Query param:** `path` (absolute or project-relative)

**Validates:** that the resolved path stays under `PROJECT_ROOT`

**Returns:** a `FileResponse`

Useful for the frontend dashboard to display:
- `agnived_cover_analysis.png`
- `vegetation_mask.tif`
- BigEarth PNGs, etc.

**Example:**
```
GET /files/image?path=Final_Res_DW/agnived_cover_analysis.png
```

#### 6.3.5. `/pipeline/run` (POST)

Runs both land-cover and vegetation sequences in order.

**Input:** `PipelineRequest`

**Steps:**

1. Calls `/landcover/dw` logic:
   - `DW_AOIConfig`, `DownloadConfig`, `run_landcover_pipeline`
2. Finds or infers `vegetation_mask.tif` path
3. Calls `run_bigearth_rdnet` with AOI in meters and `veg_mask_path`

**Returns combined JSON:**
```json
{
  "status": "success",
  "landcover": {...},
  "vegetation": {...}
}
```

### 6.4. Running the API

From the repo root:

```bash
cd "Python Backend"
python main_api.py
```

This runs:
```python
uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
```

API will be available at `http://localhost:8000`

**Interactive docs:** `http://localhost:8000/docs`

---

## 7. How Everything Fits Together

### 7.1. Typical land-cover + vegetation workflow

1. **User picks AOI on frontend:**
   - Center `(lon, lat)`, buffer (km)

2. **Backend `/landcover/dw`:**
   - Runs DW + S2 pipeline
   - Stores all outputs under `Final_Res_DW`

3. **Frontend:**
   - Uses `/files/image` to fetch:
     - `agnived_cover_analysis.png` (6-panel view)
     - Selected class masks
     - Vegetation mask
   - Displays them on a map overlay

4. **Backend `/vegetation/bigearth` or `/pipeline/run`:**
   - Runs BigEarthNet rdnet S2 classifier on a ~1.2 km patch around AOI
   - Optionally uses DW vegetation mask to focus on vegetated areas

5. **Frontend:**
   - Shows vegetation panel (BigEarth PNG) and class distribution

### 7.2. Wildlife / YouTube workflow

1. **User chooses a YouTube livestream or video in the UI**

2. **Frontend posts to `/video/classify` with `youtube_url`**

3. **Backend:**
   - Starts `run_youtube_live_inference` in background
   - Opens a local OpenCV window with annotated frames (local prototype)

4. **Future enhancements:**
   - Instead of local window, send frames or detection metadata back to frontend via WebSocket or other streaming approach

---

## 8. Notes and Assumptions

- **Earth Engine project:** `our-lamp-465108-a9` is used in all EE initialization calls

- **Band scaling:**  
  Sentinel-2 SR values (0–10000 or reflectance) are assumed numerically compatible with reBEN DN normalization. This matches the reference scripts.

- **AOI sizes:**
  - **Land-cover:** AOI radius (km) is flexible
  - **BigEarth:** minimum ≈1200 m × 1200 m to ensure at least one 120×120 tile

- **YouTube streaming:**
  - Uses `yt_dlp` to resolve and open one MP4 stream (≤720p)
  - Observed warnings (JS runtime, connection reuse) are acceptable for prototype

- **Local only:**  
  The video inference is currently designed for local desktop use (display window on the backend machine)

---

## 9. Extensibility

Planned or easy future extensions:

1. **Temporal options for land-cover and BigEarth:**
   - Expose date ranges as user-selectable in frontend

2. **Multiple AOI tiles:**
   - Extend BigEarth logic to grid over large AOIs

3. **Prithvi + SAR/optical fusion:**
   - Build on `Hyperspectral models/Prithvi model/` + `TestBigEarthS1S2.py`

4. **WebSocket-based video streaming:**
   - Replace local OpenCV window with server-side processing and client-side display

5. **Authentication & user data:**
   - Store user-submitted fauna images and metadata in DB
   - Add auth around the API

---

## Contact & Support

For issues or questions, please refer to the project documentation or contact the development team.
