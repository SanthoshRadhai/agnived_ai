from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import requests
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ee

# -----------------------------------------------------------------------------#
# Local reBEN / BigEarth metadata (copied from Test_Satellite/TestBigEarthrdnet) #
# -----------------------------------------------------------------------------#

# 10 Sentinel‑2 bands, order as used by reBEN rdnet S2 model v0.2.0
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
PATCH_SIZE = 120

BEN_CLASSES = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]

# DN‑space stats for the 10 S2 bands (same as in TestBigEarthrdnet.py)
BEN_MEAN = np.array(
    [
        1370.19151926,
        1184.38246250,
        1120.77120066,
        1136.26026392,
        1263.73947144,
        1645.40315151,
        1846.87040806,
        1762.59530783,
        1972.62420416,
        582.72633433,
    ],
    dtype=np.float32,
)
BEN_STD = np.array(
    [
        633.15169573,
        650.28427720,
        712.12507725,
        965.23119807,
        948.98199320,
        1108.06650639,
        1258.36394548,
        1233.14922810,
        1364.38688993,
        472.37967789,
    ],
    dtype=np.float32,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
# This temp dir is under the backend folder instead of /tmp
DEFAULT_TEMP_DIR = SCRIPT_DIR / "_tmp_bigearth_rdnet"
DEFAULT_TEMP_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------#
# Dataclasses                                                                  #
# -----------------------------------------------------------------------------#
@dataclass
class AOIConfig:
    lon: float
    lat: float
    # Default 0.6 km (~600 m) which yields approx 1.2 km x 1.2 km box at 10 m resolution.
    buffer_km: float = 0.6


@dataclass
class BigEarthConfig:
    temp_dir: Path = DEFAULT_TEMP_DIR
    scale: int = 10
    cloud_cover_max: int = 20
    date_start: str = "2024-01-01"
    date_end: str = "2024-11-15"


@dataclass
class BigEarthResult:
    aoi: AOIConfig
    cube_path: Path
    viz_path: Path
    class_distribution: Dict[str, float]
    tile_counts: Dict[str, int]
    avg_confidence: float
    tiles_shape: Tuple[int, int]


# -----------------------------------------------------------------------------#
# Earth Engine & model utilities                                               #
# -----------------------------------------------------------------------------#
def init_earth_engine(project: str | None = "our-lamp-465108-a9") -> None:
    """Initialize Earth Engine, mirroring the test scripts."""
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()


def load_reben_model(device: torch.device) -> torch.nn.Module:
    """
    Load the rdnet_base-s2-v0.2.0 model from the local reBEN code.

    Expects the reBEN repo (with reben_publication/) to live next to this file,
    e.g.:
        Python Backend/
          S2 Vegetation Classification pipeline/
            Vegetation_Classification_pipeline.py
            reben_publication/
    or in the project root as Test_Satellite/reben/.
    """
    import sys

    # 1) Try local "reben_publication" next to this script
    local_reben = SCRIPT_DIR / "reben_publication"
    candidate_paths = []

    if local_reben.exists():
        candidate_paths.append(local_reben)

    # 2) Fallback: original Test_Satellite/reben path (for compatibility)
    legacy_reben = PROJECT_DIR / "Test_Satellite" / "reben"
    if legacy_reben.exists():
        candidate_paths.append(legacy_reben)

    if not candidate_paths:
        raise SystemExit(
            "❌ reBEN code not found.\n"
            f"Looked for:\n"
            f"  - {local_reben}\n"
            f"  - {legacy_reben}\n"
            "Place the reBEN repo so that it contains 'reben_publication/BigEarthNetv2_0_ImageClassifier.py'."
        )

    # Use the first existing path
    reben_dir = candidate_paths[0]
    sys.path.insert(0, str(reben_dir))

    try:
        from reben_publication.BigEarthNetv2_0_ImageClassifier import (
            BigEarthNetv2_0_ImageClassifier,
        )
    except ImportError as exc:
        raise SystemExit(f"❌ Failed to import reBEN modules from {reben_dir}: {exc}")

    model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
        "BIFOLD-BigEarthNetv2-0/rdnet_base-s2-v0.2.0",
        map_location=("cuda:0" if device.type == "cuda" else "cpu"),
    )
    model = model.to(device)
    model.eval()
    return model


# -----------------------------------------------------------------------------#
# GEE composite + download                                                     #
# -----------------------------------------------------------------------------#
def build_aoi(cfg: AOIConfig) -> ee.Geometry:
    point = ee.Geometry.Point((cfg.lon, cfg.lat))
    # buffer_km -> meters
    return point.buffer(cfg.buffer_km * 1000).bounds()


def build_single_composite(aoi: ee.Geometry, cfg: BigEarthConfig) -> ee.Image:
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(cfg.date_start, cfg.date_end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.cloud_cover_max))
        .select(S2_BANDS)
    )
    if collection.size().getInfo() == 0:
        raise RuntimeError("No Sentinel‑2 scenes for the requested period.")
    return collection.median().clip(aoi)


def download_composite(img: ee.Image, aoi: ee.Geometry, cfg: BigEarthConfig) -> Path:
    cfg.temp_dir.mkdir(parents=True, exist_ok=True)
    out_tif = cfg.temp_dir / "bigearth_s2_stack.tif"

    url = img.getDownloadURL(
        {
            "scale": cfg.scale,
            "region": aoi,
            "filePerBand": False,
            "format": "GEO_TIFF",
        }
    )
    with requests.get(url, stream=True) as resp, open(out_tif, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return out_tif


def read_cube(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read()
    if data.shape[0] != len(S2_BANDS):
        raise ValueError(
            f"Expected {len(S2_BANDS)} bands, found {data.shape[0]} in {path}"
        )
    return data.astype(np.float32)


# -----------------------------------------------------------------------------#
# Tiling + normalization                                                       #
# -----------------------------------------------------------------------------#
def tile_cube(
    cube: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], int, int]:
    _, H, W = cube.shape
    h_tiles = H // PATCH_SIZE
    w_tiles = W // PATCH_SIZE
    if h_tiles == 0 or w_tiles == 0:
        raise RuntimeError("AOI too small to form a 120×120 patch.")
    tiles: List[np.ndarray] = []
    coords: List[Tuple[int, int, int, int]] = []
    for i in range(h_tiles):
        for j in range(w_tiles):
            y, x = i * PATCH_SIZE, j * PATCH_SIZE
            tiles.append(cube[:, y : y + PATCH_SIZE, x : x + PATCH_SIZE])
            coords.append((i, j, y, x))
    return np.stack(tiles), coords, h_tiles, w_tiles


def normalize_tiles(tiles: np.ndarray) -> np.ndarray:
    # BN stats are per-band; broadcast to H,W
    return (tiles - BEN_MEAN[:, None, None]) / BEN_STD[:, None, None]


def read_mask(path: Path, target_shape: Tuple[int, int]) -> np.ndarray:
    with rasterio.open(path) as src:
        m = src.read(1)
    if m.shape != target_shape:
        raise ValueError(
            f"Mask shape {m.shape} does not match cube spatial shape {target_shape}"
        )
    return m.astype(bool)


def apply_vegetation_mask(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out non‑vegetation pixels in the cube."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    masked = cube.copy()
    masked[:, ~mask] = 0.0
    return masked


# -----------------------------------------------------------------------------#
# Main pipeline wrapper                                                        #
# -----------------------------------------------------------------------------#
def run_bigearth_rdnet(
    aoi_cfg: AOIConfig,
    be_cfg: BigEarthConfig | None = None,
    device: torch.device | None = None,
    out_dir: Path | None = None,
    veg_mask_path: Path | None = None,
) -> BigEarthResult:
    if be_cfg is None:
        be_cfg = BigEarthConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if out_dir is None:
        out_dir = SCRIPT_DIR / "Results"
    out_dir.mkdir(parents=True, exist_ok=True)

    init_earth_engine()
    aoi = build_aoi(aoi_cfg)
    img = build_single_composite(aoi, be_cfg)
    cube_path = download_composite(img, aoi, be_cfg)

    s2_cube = read_cube(cube_path)
    _, H, W = s2_cube.shape

    mask = None
    if veg_mask_path is not None:
        veg_mask_path = Path(veg_mask_path)
        mask = read_mask(veg_mask_path, (H, W))
        s2_cube_masked = apply_vegetation_mask(s2_cube, mask)
    else:
        s2_cube_masked = s2_cube

    tiles, _, h_tiles, w_tiles = tile_cube(s2_cube_masked)
    tiles_norm = normalize_tiles(tiles)

    model = load_reben_model(device)
    with torch.no_grad():
        tensor = torch.from_numpy(tiles_norm).float().to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    top1_idx = np.argmax(probs, axis=1)
    top1_scores = probs[np.arange(probs.shape[0]), top1_idx]

    class_map = top1_idx.reshape(h_tiles, w_tiles)
    conf_map = top1_scores.reshape(h_tiles, w_tiles)

    counts = np.bincount(class_map.flatten(), minlength=len(BEN_CLASSES))
    total = counts.sum()
    tile_counts = {
        BEN_CLASSES[i]: int(counts[i])
        for i in range(len(BEN_CLASSES))
        if counts[i] > 0
    }
    class_distribution = (
        {k: 100.0 * v / total for k, v in tile_counts.items()} if total > 0 else {}
    )

    # Original RGB from unmasked cube
    rgb_orig = np.stack([s2_cube[2], s2_cube[1], s2_cube[0]], axis=-1)
    rgb_orig = np.clip(rgb_orig / 3000.0, 0, 1)

    # Masked RGB (dim non‑veg) if mask is provided
    if mask is not None:
        rgb_masked = rgb_orig.copy()
        # dim everything outside vegetation
        rgb_masked[~mask] *= 0.2
    else:
        rgb_masked = rgb_orig

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    axes[0].imshow(rgb_orig)
    axes[0].set_title("Sentinel‑2 True Color (original)")
    axes[0].axis("off")

    axes[1].imshow(rgb_masked)
    axes[1].set_title(
        "Sentinel‑2 True Color (vegetation highlighted)"
        if mask is not None
        else "Sentinel‑2 True Color (no mask)"
    )
    axes[1].axis("off")

    # Class map; if mask is present, treat tiles whose mean mask==0 as "Other regions"
    class_map_viz = class_map.copy()
    legend_labels = list(BEN_CLASSES)
    if mask is not None:
        tiles_mask, _, _, _ = tile_cube(mask[None, :, :].astype(float))
        tile_is_other = tiles_mask.mean(axis=(1, 2, 3)) < 0.1  # <10% veg in tile
        class_map_viz = class_map_viz.reshape(-1)
        class_map_viz[tile_is_other] = len(BEN_CLASSES)  # index for "Other regions"
        class_map_viz = class_map_viz.reshape(h_tiles, w_tiles)
        legend_labels.append("Other regions (unmasked)")

    im_class = axes[2].imshow(
        class_map_viz,
        cmap="tab20",
        vmin=0,
        vmax=len(legend_labels) - 1,
    )
    axes[2].set_title("BigEarthNet Top‑1 Class (RdNet S2)")
    axes[2].axis("off")

    legend_handles = [
        mpatches.Patch(
            color=plt.cm.tab20(c / max(1, len(legend_labels) - 1)),
            label=legend_labels[c],
        )
        for c in np.unique(class_map_viz)
    ]
    axes[2].legend(
        handles=legend_handles,
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
        fontsize=8,
    )

    im_conf = axes[3].imshow(conf_map, cmap="RdYlGn", vmin=0, vmax=1)
    axes[3].set_title("Confidence")
    axes[3].axis("off")
    plt.colorbar(im_conf, ax=axes[3], fraction=0.046, pad=0.04)

    viz_path = out_dir / "bigearth_rdnet_s2_results.png"
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150)
    plt.close()

    return BigEarthResult(
        aoi=aoi_cfg,
        cube_path=cube_path,
        viz_path=viz_path,
        class_distribution=class_distribution,
        tile_counts=tile_counts,
        avg_confidence=float(conf_map.mean()),
        tiles_shape=(h_tiles, w_tiles),
    )


if __name__ == "__main__":
    # Use buffer_km (kilometers) to be consistent with the landcover pipeline
    cfg = AOIConfig(lon=77.303778, lat=28.560278, buffer_km=3.0)

    # adjust this to wherever Download_Classify writes vegetation_mask.tif
    mask_path = Path(
        r"D:\FullStack Projects\Agnived\Agnived_transfer\Agnived_transfer\Agnived\Final_Res_DW\vegetation_mask.tif"
    )

    res = run_bigearth_rdnet(cfg, veg_mask_path=mask_path)
    print("AOI:", res.aoi)
    print("Cube:", res.cube_path)
    print("Viz:", res.viz_path)
    print("Classes:")
    for k, v in sorted(
        res.class_distribution.items(), key=lambda kv: kv[1], reverse=True
    ):
        print(f"  {k:<65} {v:5.1f}% ({res.tile_counts[k]} tiles)")
    print(f"Average confidence: {res.avg_confidence:.1%}")