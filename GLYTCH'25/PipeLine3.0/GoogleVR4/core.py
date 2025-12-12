#!/usr/bin/env python3
"""
Find well-distributed Mapillary 360° images within a circular area of interest,
and split each 360° equirectangular image into 4 non-overlapping normal images
(front/right/back/left).

The script ensures sampled images are not clustered together by maintaining
a minimum distance between selected images.

Usage:
    export MAPILLARY_TOKEN=MLY|your_token_here
    python core.py --lat 37.4219999 --lng -122.0840575 --count 5 --area-of-interest 200 --min-distance 20

Parameters:
    --lat: Center latitude
    --lng: Center longitude  
    --count: Number of images to sample (default: 3)
    --area-of-interest: Radius in meters defining the circular search area (default: 100)
    --min-distance: Minimum distance in meters between sampled images (default: 10)

Outputs:
    - Downloads panos to: out/panos/<image_id>.jpg
    - Saves 4 views each in: out/views/<image_id>_front.jpg, ..._right.jpg, ...
    - Prints JSON summary to stdout.
"""

import os
import math
import json
import argparse
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import cv2
import numpy as np
from dotenv import load_dotenv

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
OUT_DIR = BASE_DIR / "out"
PANOS_DIR = OUT_DIR / "panos"
VIEWS_DIR = OUT_DIR / "views"
PANOS_DIR.mkdir(parents=True, exist_ok=True)
VIEWS_DIR.mkdir(parents=True, exist_ok=True)


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
        "fields": "id,geometry,is_pano,thumb_2048_url,width,height",
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

        results.append(
            {
                "id": img["id"],
                "lat": img_lat,
                "lon": img_lon,
                "distance_m": dist,
                "viewer_url": f"https://www.mapillary.com/app/?focus=photo&pKey={img['id']}",
                "is_pano": is_pano,
                "thumb_2048_url": img.get("thumb_2048_url"),
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
    image_id: str, image_url: str, out_views_dir: Path
) -> Dict[str, str]:
    """
    Download the pano (if not cached), generate 4 non-overlapping perspective views
    (front/right/back/left), save them, and return dict name -> file path (string).
    """
    # Ensure output directories exist
    PANOS_DIR.mkdir(parents=True, exist_ok=True)
    out_views_dir.mkdir(parents=True, exist_ok=True)

    # Download pano if needed
    pano_path = PANOS_DIR / f"{image_id}.jpg"
    if not pano_path.exists():
        print(f"Downloading panorama {image_id}...")
        download_image(image_url, pano_path)
    else:
        print(f"Using cached panorama {image_id}")

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
        if not img["is_pano"]:
            result_payload["images"].append(
                {
                    "id": img["id"],
                    "lat": img["lat"],
                    "lon": img["lon"],
                    "distance_m": img["distance_m"],
                    "viewer_url": img["viewer_url"],
                    "is_pano": False,
                    "pano_downloaded": False,
                    "views": {},
                }
            )
            continue

        image_id = img["id"]
        image_url = img["thumb_2048_url"]
        if not image_url:
            result_payload["images"].append(
                {
                    "id": image_id,
                    "lat": img["lat"],
                    "lon": img["lon"],
                    "distance_m": img["distance_m"],
                    "viewer_url": img["viewer_url"],
                    "is_pano": True,
                    "pano_downloaded": False,
                    "views": {},
                }
            )
            continue

        try:
            views = generate_4_views_for_pano(image_id, image_url, VIEWS_DIR)
            result_payload["images"].append(
                {
                    "id": image_id,
                    "lat": img["lat"],
                    "lon": img["lon"],
                    "distance_m": img["distance_m"],
                    "viewer_url": img["viewer_url"],
                    "is_pano": True,
                    "pano_downloaded": True,
                    "pano_path": str(PANOS_DIR / f"{image_id}.jpg"),
                    "views": views,  # dict: front/right/back/left -> file path
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
                    "is_pano": True,
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