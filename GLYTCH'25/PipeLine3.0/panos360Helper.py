
import os
import math
import json
import argparse
import time
import traceback
from typing import Optional, Dict, Any, List
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import cv2
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
import os

import torch
from PIL import Image
# add to the existing imports
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoImageProcessor,
    AutoModelForImageClassification,
)
import asyncio
from pydantic import Field

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


PLANT_MODEL_ID = "juppy44/plant-identification-2m-vit-b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plant_processor = AutoImageProcessor.from_pretrained(PLANT_MODEL_ID)
plant_model = AutoModelForImageClassification.from_pretrained(PLANT_MODEL_ID).to(DEVICE)


app = FastAPI(title="AgniVed Pipeline API", version="1.0.0")

class LandcoverVegetationPanosRequest(BaseModel):
    lon: float
    lat: float
    buffer_km: float = 3.1
    date_start: str = "2024-10-01"
    date_end: str = "2024-11-15"
    scale: int = 10
    cloud_cover_max: int = 20
    panos_lat: Optional[float] = None
    panos_lon: Optional[float] = None
    panos_count: int = 3
    panos_area_of_interest: float = 100.0
    panos_min_distance: float = 20.0
    panos_labels: List[str] = ["tree", "bushes"]



# ---------------------------------------------------------------------
# Config / env
# ---------------------------------------------------------------------
load_dotenv()

MAPILLARY_TOKEN_ENV = MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN")

if not MAPILLARY_TOKEN:
    raise RuntimeError(
        f"{MAPILLARY_TOKEN_ENV} is not set. "
        "Set it to your Mapillary access token, e.g.\n"
        "  export MAPILLARY_TOKEN=MLY|your_token_here"
    )

MAPILLARY_IMAGES_URL = "https://graph.mapillary.com/images"

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "Pano360OutputDir"
PANOS_DIR = OUT_DIR / "panos"
VIEWS_DIR = OUT_DIR / "views"
DETECTED_CROPS_DIR = OUT_DIR / "detected_crops"
PANOS_DIR.mkdir(parents=True, exist_ok=True)
VIEWS_DIR.mkdir(parents=True, exist_ok=True)
DETECTED_CROPS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# HTTP Session with retry logic
# ---------------------------------------------------------------------
def create_session_with_retries(
    retries: int = 3,
    backoff_factor: float = 0.5,
    timeout: int = 30
) -> requests.Session:
    """
    Create a requests session with automatic retry logic.
    
    Args:
        retries: Number of retries for failed requests
        backoff_factor: Sleep time between retries (backoff_factor * (2 ** retry_number))
        timeout: Default timeout for requests
    
    Returns:
        Configured requests.Session object
    """
    session = requests.Session()
    
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


# ---------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------
def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distance between two (lat, lon) points in meters (haversine formula).
    """
    R = 6371000.0  # Earth radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def bbox_from_point(
    lat: float, lon: float, radius_m: float = 100.0
) -> Dict[str, float]:
    """
    Approximate bounding box (in degrees) around (lat, lon) for given radius (meters).
    Returns dict with min_lat, max_lat, min_lon, max_lon.
    """
    # 1 degree latitude ≈ 111.32 km
    delta_lat = radius_m / 111_320.0

    # 1 degree longitude ≈ 111.32 km * cos(latitude)
    lat_rad = math.radians(lat)
    meters_per_deg_lon = 111_320.0 * max(math.cos(lat_rad), 1e-6)
    delta_lon = radius_m / meters_per_deg_lon

    return {
        "min_lat": lat - delta_lat,
        "max_lat": lat + delta_lat,
        "min_lon": lon - delta_lon,
        "max_lon": lon + delta_lon,
    }


# ---------------------------------------------------------------------
# Spatial distribution helpers
# ---------------------------------------------------------------------
def select_well_distributed_images(
    images: List[Dict[str, Any]],
    n: int,
    min_distance_m: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Select n well-distributed images from the list, ensuring they are not too close to each other.
    Uses a greedy algorithm to maximize minimum distance between selected images.
    
    Args:
        images: List of image dictionaries with 'lat', 'lon', 'distance_m' fields
        n: Number of images to select
        min_distance_m: Minimum distance (in meters) between selected images
    
    Returns:
        List of selected images, well-distributed across the area
    """
    if len(images) <= n:
        return images
    
    # Start with the closest image to the center
    selected = [images[0]]
    remaining = images[1:]
    
    while len(selected) < n and remaining:
        # Find the image that maximizes the minimum distance to all selected images
        best_candidate = None
        best_min_dist = 0
        best_idx = -1
        
        for idx, candidate in enumerate(remaining):
            # Calculate minimum distance to any already-selected image
            min_dist_to_selected = min(
                haversine_distance_m(
                    candidate["lat"], candidate["lon"],
                    sel["lat"], sel["lon"]
                )
                for sel in selected
            )
            
            # Keep track of the candidate with the largest minimum distance
            if min_dist_to_selected > best_min_dist:
                best_min_dist = min_dist_to_selected
                best_candidate = candidate
                best_idx = idx
        
        # If the best candidate is too close (less than min_distance_m), stop
        if best_min_dist < min_distance_m and len(selected) > 0:
            print(f"  Note: Only found {len(selected)} images with min separation of {min_distance_m}m")
            break
        
        # Add the best candidate to selected images
        if best_candidate:
            selected.append(best_candidate)
            remaining.pop(best_idx)
    
    return selected


# ---------------------------------------------------------------------
# Mapillary: get images within area of interest
# ---------------------------------------------------------------------
def find_nearest_vr_images(
    lat: float,
    lon: float,
    n: int = 5,
    radius_m: float = 100.0,
    min_distance_m: float = 10.0,
    token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Query Mapillary v4 for well-distributed images within a circular area of interest.

    Args:
        lat: Center latitude
        lon: Center longitude
        n: Number of images to return
        radius_m: Area of interest radius in meters (search within this circle)
        min_distance_m: Minimum distance between selected images in meters
        token: Mapillary API token
        
    Returns a list of dicts:
      [
        {
          "id": <image_id>,
          "lat": <image_lat>,
          "lon": <image_lon>,
          "distance_m": <distance>,
          "viewer_url": "...",
          "is_pano": <bool>,
          "thumb_2048_url": <str | None>
        },
        ...
      ]
    Sorted by distance_m ascending, with good spatial distribution.
    """
    if token is None:
        token = MAPILLARY_TOKEN

    bbox = bbox_from_point(lat, lon, radius_m=radius_m)

    params = {
        # Ask for everything we need in one call
        # thumb_original_url is the full-resolution image URL for non-panos
        "fields": "id,geometry,is_pano,thumb_2048_url,thumb_original_url,width,height",
        "bbox": f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}",
        "limit": 2000,  # upper bound; API may cap it
    }
    headers = {"Authorization": f"OAuth {token}"}

    # Create session with retry logic
    session = create_session_with_retries(retries=3, backoff_factor=1.0, timeout=30)
    
    try:
        print(f"Querying Mapillary API for images within {radius_m}m of ({lat}, {lon})...")
        resp = session.get(MAPILLARY_IMAGES_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        print(f"API request successful. Processing results...")
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out. Please check your internet connection.")
        print("You can try:")
        print("  1. Increasing the radius with --area-of-interest parameter")
        print("  2. Checking your network connection")
        print("  3. Trying again later")
        return []
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Mapillary API.")
        print("Please check your internet connection and firewall settings.")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error occurred: {e}")
        if resp.status_code == 401:
            print("Authentication failed. Please check your MAPILLARY_TOKEN.")
        elif resp.status_code == 403:
            print("Access forbidden. Your token may not have the required permissions.")
        return []
    except Exception as e:
        print(f"ERROR: Unexpected error while querying API: {e}")
        return []
    finally:
        session.close()

    images = data.get("data") or []
    if not images:
        return []

    results: List[Dict[str, Any]] = []
    for img in images:
        geom = img.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue

        img_lon, img_lat = coords[0], coords[1]
        dist = haversine_distance_m(lat, lon, img_lat, img_lon)
        
        # Filter: only include images within the circular area of interest
        if dist > radius_m:
            continue

        width = img.get("width")
        height = img.get("height")
        aspect = None
        if width and height and height != 0:
            aspect = width / height

        # "is_pano" flag from API; if missing, try aspect ratio heuristics
        is_pano = bool(img.get("is_pano", False))
        if not is_pano and aspect is not None and abs(aspect - 2.0) < 0.3:
            is_pano = True

        # Get image URL - prefer thumb_2048_url for panos, thumb_original_url for non-panos
        image_url = img.get("thumb_2048_url") or img.get("thumb_original_url")

        results.append(
            {
                "id": img["id"],
                "lat": img_lat,
                "lon": img_lon,
                "distance_m": dist,
                "viewer_url": f"https://www.mapillary.com/app/?focus=photo&pKey={img['id']}",
                "is_pano": is_pano,
                "thumb_2048_url": img.get("thumb_2048_url"),
                "thumb_original_url": img.get("thumb_original_url"),
                "image_url": image_url,
                "width": width,
                "height": height,
                "aspect": aspect,
            }
        )

    # Sort by distance from center
    results.sort(key=lambda r: r["distance_m"])
    
    print(f"Found {len(results)} total images in the area of interest")
    
    # Select well-distributed samples
    if len(results) > n:
        print(f"Selecting {n} well-distributed images (min {min_distance_m}m apart)...")
        selected = select_well_distributed_images(results, n, min_distance_m)
        return selected
    
    return results


# ---------------------------------------------------------------------
# 360 (equirectangular) → 4 normal views
# ---------------------------------------------------------------------
def perspective_from_equirect(
    pano: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """
    Convert equirectangular pano to a single perspective view.

    pano: H x W x 3 BGR
    yaw_deg: yaw angle in degrees
    pitch_deg: pitch angle in degrees (positive = look up)
    fov_deg: horizontal field of view in degrees
    out_w, out_h: output resolution of the perspective image
    """
    h_in, w_in = pano.shape[:2]
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    fov = np.deg2rad(fov_deg)

    # focal length in pixels
    f = 0.5 * out_w / math.tan(fov / 2.0)

    xs = np.linspace(-out_w / 2.0, out_w / 2.0, out_w)
    ys = np.linspace(-out_h / 2.0, out_h / 2.0, out_h)
    x_grid, y_grid = np.meshgrid(xs, ys)

    z = f * np.ones_like(x_grid)
    x = x_grid
    y = -y_grid  # flip Y to match image coordinates

    # normalize direction vectors
    norm = np.sqrt(x * x + y * y + z * z)
    x /= norm
    y /= norm
    z /= norm

    # pitch around X-axis
    sin_pitch, cos_pitch = np.sin(pitch), np.cos(pitch)
    y_p = y * cos_pitch - z * sin_pitch
    z_p = y * sin_pitch + z * cos_pitch
    x_p = x

    # yaw around Y-axis
    sin_yaw, cos_yaw = np.sin(yaw), np.cos(yaw)
    x_y = x_p * cos_yaw + z_p * sin_yaw
    y_y = y_p
    z_y = -x_p * sin_yaw + z_p * cos_yaw

    # convert to spherical (lon, lat)
    lon = np.arctan2(x_y, z_y)  # [-pi, pi]
    lat = np.arcsin(y_y)        # [-pi/2, pi/2]

    # spherical → equirectangular pixel coords
    x_pano = (lon / (2 * np.pi) + 0.5) * w_in
    y_pano = (0.5 - lat / np.pi) * h_in

    map_x = x_pano.astype(np.float32)
    map_y = y_pano.astype(np.float32)

    perspective = cv2.remap(
        pano, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    )
    return perspective


def download_image(url: str, out_path: Path) -> None:
    """
    Download image from URL to out_path with retry logic.
    """
    session = create_session_with_retries(retries=3, backoff_factor=1.0, timeout=30)
    
    try:
        r = session.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"  Downloaded: {out_path.name}")
    except requests.exceptions.Timeout:
        print(f"  ERROR: Timeout downloading {out_path.name}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Failed to download {out_path.name}: {e}")
        raise
    finally:
        session.close()


def generate_4_views_for_pano(
    image_id: str, image_url: str, out_views_dir: Path, is_pano: bool = True
) -> Dict[str, str]:
    """
    Download the image (if not cached). If it's a pano, generate 4 non-overlapping perspective views
    (front/right/back/left). If it's not a pano, just save the original image as a single view.
    Returns dict name -> file path (string).
    """
    import shutil
    
    # Ensure output directories exist
    PANOS_DIR.mkdir(parents=True, exist_ok=True)
    out_views_dir.mkdir(parents=True, exist_ok=True)

    # Download image if needed
    pano_path = PANOS_DIR / f"{image_id}.jpg"
    if not pano_path.exists():
        print(f"Downloading image {image_id}...")
        download_image(image_url, pano_path)
    else:
        print(f"Using cached image {image_id}")

    # If not a panorama, just return the original image as a single view
    if not is_pano:
        print(f"  Image {image_id} is not a panorama, using as-is")
        out_path = out_views_dir / f"{image_id}_original.jpg"
        # Copy the image to views directory
        shutil.copy(str(pano_path), str(out_path))
        return {"original": str(out_path)}

    # For panoramas, generate 4 perspective views
    pano = cv2.imread(str(pano_path))
    if pano is None:
        print(f"  ERROR: Could not load image {pano_path}")
        return {}

    print(f"  Generating 4 perspective views...")
    out_w, out_h = 800, 600
    fov_deg = 90.0
    pitch_deg = 0.0
    yaw_list = [0, 90, 180, 270]
    label_map = {0: "front", 90: "right", 180: "back", 270: "left"}

    out: Dict[str, str] = {}

    for yaw in yaw_list:
        view = perspective_from_equirect(
            pano,
            yaw_deg=yaw,
            pitch_deg=pitch_deg,
            fov_deg=fov_deg,
            out_w=out_w,
            out_h=out_h,
        )
        name = label_map.get(yaw, str(yaw))
        out_path = out_views_dir / f"{image_id}_{name}.jpg"
        cv2.imwrite(str(out_path), view)
        out[name] = str(out_path)

    print(f"  Generated: {', '.join(out.keys())}")
    return out


# ---------------------------------------------------------------------
# Main script logic
# ---------------------------------------------------------------------

# Flask-callable function
def find_panos_and_views(lat, lon, n=3, radius_m=100.0, min_distance_m=10.0):
    images = find_nearest_vr_images(lat, lon, n=n, radius_m=radius_m, min_distance_m=min_distance_m)
    if not images:
        return {
            "found": False,
            "message": "No imagery found in given area of interest.",
            "lat": lat,
            "lon": lon,
            "area_of_interest_m": radius_m,
            "min_distance_m": min_distance_m,
        }

    result_payload = {
        "found": True,
        "input_lat": lat,
        "input_lon": lon,
        "area_of_interest_m": radius_m,
        "min_distance_m": min_distance_m,
        "count_requested": n,
        "count_returned": len(images),
        "images": [],
    }

    for img in images:
        image_id = img["id"]
        image_url = img.get("image_url") or img.get("thumb_2048_url") or img.get("thumb_original_url")
        is_pano = img["is_pano"]
        
        if not image_url:
            result_payload["images"].append(
                {
                    "id": image_id,
                    "lat": img["lat"],
                    "lon": img["lon"],
                    "distance_m": img["distance_m"],
                    "viewer_url": img["viewer_url"],
                    "is_pano": is_pano,
                    "pano_downloaded": False,
                    "views": {},
                }
            )
            continue

        try:
            # Download and process image (whether pano or not)
            views = generate_4_views_for_pano(image_id, image_url, VIEWS_DIR, is_pano=is_pano)
            result_payload["images"].append(
                {
                    "id": image_id,
                    "lat": img["lat"],
                    "lon": img["lon"],
                    "distance_m": img["distance_m"],
                    "viewer_url": img["viewer_url"],
                    "is_pano": is_pano,
                    "pano_downloaded": True,
                    "pano_path": str(PANOS_DIR / f"{image_id}.jpg"),
                    "views": views,  # dict: front/right/back/left -> file path (or "original" for non-panos)
                }
            )
        except Exception as e:
            result_payload["images"].append(
                {
                    "id": image_id,
                    "lat": img["lat"],
                    "lon": img["lon"],
                    "distance_m": img["distance_m"],
                    "viewer_url": img["viewer_url"],
                    "is_pano": is_pano,
                    "pano_downloaded": False,
                    "views": {},
                    "error": str(e),
                }
            )

    return result_payload


# # Larger area, more samples, closer together
# python .\core.py --lat 37.4219999 --lng -122.0840575 --count 10 --area-of-interest 1000 --min-distance 30

# # Small area, few samples, well-spread
# python .\core.py --lat 24.492786100000018 --lng 77.34341670000003 --count 3 --area-of-interest 200 --min-distance 100

# # Using short form parameters
# python .\core.py --lat 24.4928 --lng 77.3434 --count 8 --aoi 750 --min-distance 40



def run_object_detection(image_path, labels, output_dir=None, score_threshold=0.4, text_threshold=0.3):
    if output_dir is None:
        output_dir = str(DETECTED_CROPS_DIR)
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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