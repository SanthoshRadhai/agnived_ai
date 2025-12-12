"""
create_vegetation_timeseries.py

Updated: skip windows that have no Sentinel-2 or Dynamic World (no empty frames).
Also ensures all frames have identical dimensions and are padded to a multiple of 16
to avoid ffmpeg macro block size errors.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import json
import math
import sys

import ee
import geemap
import numpy as np
from PIL import Image, ImageOps
import imageio.v3 as iio
import rasterio

# --- Config dataclasses (reuse from your pipeline) ---

@dataclass
class AOIConfig:
    lon: float
    lat: float
    buffer_km: float

@dataclass
class DownloadConfig:
    output_dir: Path
    date_start: str
    date_end: str
    scale: int = 10
    cloud_cover_max: int = 20
    frames: int = 15
    veg_threshold: float = 0.27
    fps: int = 5
    ee_project: Optional[str] = "our-lamp-465108-a9"


DYNAMIC_WORLD_CLASSES = {
    0: {"name": "Water", "color": "#419BDF"},
    1: {"name": "Trees", "color": "#397D49"},
    2: {"name": "Grass", "color": "#88B053"},
    3: {"name": "Flooded Vegetation", "color": "#7A87C6"},
    4: {"name": "Crops", "color": "#E49635"},
    5: {"name": "Shrub & Scrub", "color": "#DFC35A"},
    6: {"name": "Built Area", "color": "#C4281B"},
    7: {"name": "Bare Ground", "color": "#A59B8F"},
    8: {"name": "Snow & Ice", "color": "#B39FE1"},
}

S2_BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
DW_PROBABILITY_BANDS = ["water","trees","grass","flooded_vegetation","crops","shrub_and_scrub","built","bare","snow_and_ice"]


# --- Earth Engine helpers ---

def init_ee(project: Optional[str]) -> None:
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()


def create_aoi(config: AOIConfig) -> ee.Geometry:
    center = ee.Geometry.Point([config.lon, config.lat])
    return center.buffer(config.buffer_km * 1000).bounds()


def load_sentinel2(aoi: ee.Geometry, cfg: DownloadConfig, start: str, end: str) -> ee.ImageCollection:
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.cloud_cover_max)))


def create_sentinel2_composite(collection: ee.ImageCollection) -> ee.Image:
    return collection.select(S2_BANDS).median()


def load_dynamic_world(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    return (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(aoi)
            .filterDate(start, end))


def create_probability_composite(dw_collection: ee.ImageCollection) -> ee.Image:
    return dw_collection.select(DW_PROBABILITY_BANDS).mean()


def create_vegetation_probability_image(prob_image: ee.Image) -> ee.Image:
    veg = prob_image.select(['trees','grass','crops','shrub_and_scrub','flooded_vegetation'])
    veg_max = veg.reduce(ee.Reducer.max())
    return veg_max


def download_geotiff(image: ee.Image, aoi: ee.Geometry, filename: str, cfg: DownloadConfig) -> Path:
    output_path = cfg.output_dir / filename
    geemap.download_ee_image(
        image,
        filename=str(output_path),
        scale=cfg.scale,
        region=aoi,
        crs="EPSG:4326",
    )
    return output_path


# --- Local raster helpers ---

def read_raster_as_array(path: Path) -> np.ndarray:
    with rasterio.open(str(path)) as src:
        arr = src.read(1).astype('float32')
        if arr.max() > 1.0:
            if arr.max() > 1000:
                arr = arr / 10000.0
            else:
                arr = arr / arr.max()
        arr = np.clip(arr, 0.0, 1.0)
        return arr


def save_probability_png(arr: np.ndarray, out_path: Path):
    g = np.clip(arr, 0.0, 1.0)
    r = (g * 0.1) ** 0.9
    gg = g ** 0.9
    b = (g * 0.05) ** 1.1
    rgb = np.stack([r, gg, b], axis=-1)
    img = (np.clip(rgb, 0.0, 1.0) * 255).astype('uint8')
    Image.fromarray(img).save(str(out_path))


# --- Time window utilities ---

def split_date_range(start_date: str, end_date: str, windows: int) -> List[Dict[str,str]]:
    s = datetime.fromisoformat(start_date)
    e = datetime.fromisoformat(end_date)
    total_days = (e - s).days
    if total_days < 1:
        raise ValueError("End date must be after start date")
    window_days = max(1, math.floor(total_days / windows))
    ranges = []
    cur = s
    for i in range(windows):
        start = cur
        end = start + timedelta(days=window_days - 1)
        if i == windows - 1:
            end = e
        ranges.append({"start": start.date().isoformat(), "end": end.date().isoformat()})
        cur = end + timedelta(days=1)
        if cur > e:
            cur = e
    return ranges


# --- Utilities for consistent frame size & padding --- 

def make_frame_consistent(img: Image.Image, target_size: (int, int)) -> Image.Image:
    """
    Resize (keeping aspect) and pad to target_size (w,h) with black background.
    """
    return ImageOps.fit(img, target_size, method=Image.BILINEAR, centering=(0.5, 0.5))


def pad_to_multiple_of_16(img: Image.Image) -> Image.Image:
    w, h = img.size
    target_w = ( (w + 15) // 16 ) * 16
    target_h = ( (h + 15) // 16 ) * 16
    if (target_w, target_h) == (w, h):
        return img
    # center pad
    new_img = Image.new("RGB", (target_w, target_h), (0,0,0))
    left = (target_w - w) // 2
    top = (target_h - h) // 2
    new_img.paste(img, (left, top))
    return new_img


# --- Video helper (works with imageio.v3 and falls back to legacy imageio) ---

def save_video(frame_paths: List[Path], out_path: Path, fps: int = 5, slow_factor: int = 2):
    """
    Save an MP4. Frames are normalized to same size and padded to multiple of 16.
    """
    if not frame_paths:
        raise RuntimeError("No frames available to write video (all windows were empty).")

    # Load frames and determine a target size (use max width/height among frames to avoid too much upscale)
    pil_frames = []
    max_w = 0
    max_h = 0
    for p in frame_paths:
        img = Image.open(str(p)).convert('RGB')
        pil_frames.append(img)
        w,h = img.size
        if w > max_w: max_w = w
        if h > max_h: max_h = h

    # Use that max as our target size (fit each image into it preserving aspect)
    target_size = (max_w, max_h)
    normalized = []
    for img in pil_frames:
        # fit + pad to target_size
        fitted = make_frame_consistent(img, target_size)
        padded = pad_to_multiple_of_16(fitted)
        normalized.append(np.array(padded))

    # write via v3.imwrite plugin='ffmpeg' if available, else fallback to legacy writer
    try:
        iio.imwrite(
            str(out_path),
            normalized,
            plugin='ffmpeg',
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p'
        )
        return
    except Exception as e:
        print("imageio.v3 ffmpeg write failed:", e)
        print("Attempting legacy imageio fallback (get_writer)...")

    # legacy fallback
    try:
        import imageio_ffmpeg 
        writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264')
        for frame in normalized:
            for _ in range(slow_factor):
                writer.append_data(frame)
        writer.close()
        return
    except Exception as e2:
        print("Legacy imageio fallback failed:", e2)
        print("Final failure: please install imageio-ffmpeg or add ffmpeg to PATH.")
        raise e2


# --- Pipeline ---

def process_timeseries(aoi_cfg: AOIConfig, dl_cfg: DownloadConfig) -> Dict:
    dl_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    init_ee(dl_cfg.ee_project)

    aoi = create_aoi(aoi_cfg)
    windows = split_date_range(dl_cfg.date_start, dl_cfg.date_end, dl_cfg.frames)

    frame_paths: List[Path] = []
    veg_stats = []

    for idx, w in enumerate(windows):
        print(f"Processing frame {idx+1}/{len(windows)}: {w['start']} -> {w['end']}")
        s2_coll = load_sentinel2(aoi, dl_cfg, w['start'], w['end'])
        if s2_coll.size().getInfo() == 0:
            print("  No Sentinel-2 for this window; skipping frame")
            continue

        dw_coll = load_dynamic_world(aoi, w['start'], w['end'])
        if dw_coll.size().getInfo() == 0:
            print("  No Dynamic World for this window; skipping frame")
            continue

        # Both present -> proceed
        s2_img = create_sentinel2_composite(s2_coll)
        prob_img = create_probability_composite(dw_coll)
        veg_prob_img = create_vegetation_probability_image(prob_img)

        veg_prob_tif = download_geotiff(veg_prob_img, aoi, f"veg_prob_{idx:03d}.tif", dl_cfg)
        arr = read_raster_as_array(veg_prob_tif)

        png_path = dl_cfg.output_dir / f"veg_prob_{idx:03d}.png"
        save_probability_png(arr, png_path)

        veg_mask = arr >= dl_cfg.veg_threshold
        veg_pct = float(100.0 * veg_mask.sum() / veg_mask.size)
        veg_stats.append({"start": w['start'], "end": w['end'], "veg_pct": veg_pct})

        frame_paths.append(png_path)

    if not frame_paths:
        raise RuntimeError("All windows were empty - no frames downloaded. Try a wider date range or smaller windows.")

    video_path = dl_cfg.output_dir / "vegetation_timeseries.mp4"
    print(f"Assembling video to {video_path} at {dl_cfg.fps} fps ({len(frame_paths)} frames)")

    try:
        save_video(frame_paths, video_path, fps=dl_cfg.fps, slow_factor=2)
    except Exception as e:
        print("Failed to write video. See error above.")
        raise

    stats_path = dl_cfg.output_dir / "vegetation_area_trend.json"
    with open(stats_path, 'w') as f:
        json.dump(veg_stats, f, indent=2)

    summary = {
        "video": str(video_path),
        "frames": [str(p) for p in frame_paths],
        "stats": str(stats_path),
        "veg_stats": veg_stats,
    }
    print("Done. Summary:")
    print(json.dumps(summary, indent=2))
    return summary


# --- Default run ---
if __name__ == "__main__":
    default_cfg = DownloadConfig(
        output_dir=Path("./Final_Res_DW_timeseries"),
        date_start="2023-10-01",
        date_end="2024-12-31",
        scale=10,
        cloud_cover_max=30,
        frames=15,
        veg_threshold=0.27,
        fps=2,
    )
    default_aoi = AOIConfig(lon=77.303778, lat=28.560278, buffer_km=3.0)

    process_timeseries(default_aoi, default_cfg)