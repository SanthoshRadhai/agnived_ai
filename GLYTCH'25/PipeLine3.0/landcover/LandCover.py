import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import ee
import geemap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio


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
    ee_project: Optional[str] = "our-lamp-465108-a9"


DYNAMIC_WORLD_CLASSES = {
    0: {"name": "Water", "color": "#419BDF", "description": "Permanent and seasonal water bodies"},
    1: {"name": "Trees", "color": "#397D49", "description": "Forests, tree plantations, orchards"},
    2: {"name": "Grass", "color": "#88B053", "description": "Natural grasslands, pastures, parks"},
    3: {"name": "Flooded Vegetation", "color": "#7A87C6", "description": "Mangroves, wetlands"},
    4: {"name": "Crops", "color": "#E49635", "description": "Agricultural land, croplands"},
    5: {"name": "Shrub & Scrub", "color": "#DFC35A", "description": "Shrublands, scrublands, bushes"},
    6: {"name": "Built Area", "color": "#C4281B", "description": "Urban, buildings, roads"},
    7: {"name": "Bare Ground", "color": "#A59B8F", "description": "Exposed soil, sand, rocks"},
    8: {"name": "Snow & Ice", "color": "#B39FE1", "description": "Snow cover, glaciers"},
}

S2_BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
DW_PROBABILITY_BANDS = ["water","trees","grass","flooded_vegetation","crops","shrub_and_scrub","built","bare","snow_and_ice"]


def init_ee(project: Optional[str]) -> None:
    """Initialize Earth Engine."""
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()


def create_aoi(config: AOIConfig) -> ee.Geometry:
    """Create AOI geometry from lon/lat/buffer."""
    center = ee.Geometry.Point([config.lon, config.lat])
    return center.buffer(config.buffer_km * 1000).bounds()


def load_sentinel2(aoi: ee.Geometry, cfg: DownloadConfig) -> ee.ImageCollection:
    """Load Sentinel‑2 collection for the AOI."""
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(cfg.date_start, cfg.date_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cfg.cloud_cover_max)))


def create_sentinel2_composite(collection: ee.ImageCollection) -> ee.Image:
    """Median composite of all Sentinel‑2 spectral bands."""
    return collection.select(S2_BANDS).median()


def load_dynamic_world(aoi: ee.Geometry, cfg: DownloadConfig) -> ee.ImageCollection:
    """Load Dynamic World collection for the AOI."""
    return (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(aoi)
            .filterDate(cfg.date_start, cfg.date_end))


def create_classification_composite(dw_collection: ee.ImageCollection) -> ee.Image:
    """Mode composite of Dynamic World labels."""
    return dw_collection.select("label").mode()


def create_probability_composite(dw_collection: ee.ImageCollection) -> ee.Image:
    """Mean probability composite for Dynamic World classes."""
    return dw_collection.select(DW_PROBABILITY_BANDS).mean()

def create_vegetation_mask(dw_collection, threshold=0.5):
    """Return EE image where vegetation probability ≥ threshold."""
    veg_bands = dw_collection.select(['trees','grass','crops','shrub_and_scrub','flooded_vegetation']).mean()
    veg_prob = veg_bands.reduce(ee.Reducer.max())
    return veg_prob.gte(threshold)

def download_geotiff(image: ee.Image, aoi: ee.Geometry, filename: str,
                     cfg: DownloadConfig, bands: Optional[list] = None) -> Path:
    """Download an EE image as GeoTIFF."""
    output_path = cfg.output_dir / filename
    if bands:
        image = image.select(bands)
    geemap.download_ee_image(
        image,
        filename=str(output_path),
        scale=cfg.scale,
        region=aoi,
        crs="EPSG:4326",
    )
    return output_path


def calculate_statistics(classification_path: Path, cfg: DownloadConfig) -> Dict[str, Dict]:
    """Compute coverage and area per Dynamic World class."""
    with rasterio.open(classification_path) as src:
        raster = src.read(1)
        transform = src.transform
    unique, counts = np.unique(raster, return_counts=True)
    stats = {}
    pixel_area_km2 = (cfg.scale * cfg.scale) / 1e6
    total_pixels = counts.sum()
    for class_id, count in zip(unique, counts):
        if class_id in DYNAMIC_WORLD_CLASSES:
            info = DYNAMIC_WORLD_CLASSES[class_id]
            stats[info["name"]] = {
                "pixels": int(count),
                "area_km2": float(count * pixel_area_km2),
                "percentage": float(100 * count / total_pixels),
                "description": info["description"],
            }
    stats["_meta"] = {
        "pixel_area_km2": pixel_area_km2,
        "total_pixels": int(total_pixels),
        "transform": transform[:6],
    }
    return stats



def create_visualizations(classification_path: Path,
                          probabilities_path: Path,
                          sentinel2_path: Path,
                          stats: Dict[str, Dict],
                          cfg: DownloadConfig) -> Path:
    """Create AgniVed land‑cover visualization (layout based on TestClassificationDownload.py)."""
    print("\n🎨 Creating AgniVed cover visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("AgniVed Cover Analysis", fontsize=18, fontweight="bold")

    # classification raster + extent
    with rasterio.open(classification_path) as src:
        classification_data = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    # Sentinel‑2 RGB + NIR
    with rasterio.open(sentinel2_path) as src:
        red = src.read(4)
        green = src.read(3)
        blue = src.read(2)
        nir = src.read(8)

    rgb = np.dstack([
        np.clip(red / 3000, 0, 1),
        np.clip(green / 3000, 0, 1),
        np.clip(blue / 3000, 0, 1),
    ])
    false_color = np.dstack([
        np.clip(nir / 3000, 0, 1),
        np.clip(red / 3000, 0, 1),
        np.clip(green / 3000, 0, 1),
    ])

    # 1) S2 true color
    ax1 = axes[0, 0]
    ax1.imshow(rgb, extent=extent)
    ax1.set_title("Sentinel‑2 True Color (RGB)", fontweight="bold")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # 2) Classification map with legend on the side (like TestClassificationDownload.py)
    ax2 = axes[0, 1]
    colors = [DYNAMIC_WORLD_CLASSES[i]["color"] for i in range(9)]
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    im_class = ax2.imshow(
        classification_data, cmap=cmap, vmin=0, vmax=8, extent=extent
    )
    ax2.set_title("Land Cover Classification", fontweight="bold")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    legend_handles = [
        mpatches.Patch(
            color=DYNAMIC_WORLD_CLASSES[i]["color"],
            label=DYNAMIC_WORLD_CLASSES[i]["name"],
        )
        for i in range(9)
        if i in DYNAMIC_WORLD_CLASSES
    ]
    ax2.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        borderaxespad=0.0,
    )

    cbar = fig.colorbar(im_class, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Class ID")

    # 3) Land‑cover stats bar chart (same idea as TestClassificationDownload.py)
    ax3 = axes[0, 2]
    class_names = [name for name in stats if not name.startswith("_")]
    percentages = [stats[name]["percentage"] for name in class_names]
    colors_bar = []
    for name in class_names:
        color = next(
            (v["color"] for v in DYNAMIC_WORLD_CLASSES.values() if v["name"] == name),
            "#cccccc",
        )
        colors_bar.append(color)

    bars = ax3.barh(class_names, percentages, color=colors_bar)
    ax3.set_xlabel("Coverage (%)", fontweight="bold")
    ax3.set_title("Land Cover Distribution", fontweight="bold")
    ax3.set_xlim(0, 100)

    for bar in bars:
        width = bar.get_width()
        ax3.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            fontsize=8,
        )

    # 4) S2 false color
    ax4 = axes[1, 0]
    ax4.imshow(false_color, extent=extent)
    ax4.set_title(
        "Sentinel‑2 False Color (NIR‑R‑G)\nVegetation appears red",
        fontweight="bold",
    )
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")

    # 5) Tree probability
    ax5 = axes[1, 1]
    with rasterio.open(probabilities_path) as src:
        trees_prob = src.read(2)
    im_trees = ax5.imshow(
        trees_prob, cmap="Greens", vmin=0, vmax=1, extent=extent
    )
    ax5.set_title("Tree Coverage Probability", fontweight="bold")
    ax5.set_xlabel("Longitude")
    ax5.set_ylabel("Latitude")
    fig.colorbar(im_trees, ax=ax5, label="Probability")

    # 6) Built‑area probability
    ax6 = axes[1, 2]
    with rasterio.open(probabilities_path) as src:
        built_prob = src.read(7)
    im_built = ax6.imshow(
        built_prob, cmap="Reds", vmin=0, vmax=1, extent=extent
    )
    ax6.set_title("Built Area Probability", fontweight="bold")
    ax6.set_xlabel("Longitude")
    ax6.set_ylabel("Latitude")
    fig.colorbar(im_built, ax=ax6, label="Probability")

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    viz_path = cfg.output_dir / "agnived_cover_analysis.png"
    fig.savefig(viz_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    create_class_masks(classification_data, extent, cfg)
    return viz_path


def create_class_masks(classification_data: np.ndarray, extent, cfg: DownloadConfig) -> None:
    """Generate binary masks per Dynamic World class."""
    for class_id, class_info in DYNAMIC_WORLD_CLASSES.items():
        mask = (classification_data == class_id).astype(np.uint8)
        if mask.sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(mask, cmap="gray", extent=extent)
        ax.set_title(f"{class_info['name']} Mask")
        mask_path = cfg.output_dir / f"mask_{class_info['name'].lower().replace(' ', '_')}.png"
        fig.savefig(mask_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_metadata(stats: Dict[str, Dict], aoi: ee.Geometry, cfg: DownloadConfig) -> Path:
    """Persist a metadata JSON with AOI footprint and statistics."""
    coords = aoi.coordinates().getInfo()[0]
    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "data_sources": {
            "land_cover": "Google Dynamic World V1",
            "hyperspectral": "Sentinel-2 SR Harmonized",
        },
        "resolution_m": cfg.scale,
        "date_range": {"start": cfg.date_start, "end": cfg.date_end},
        "aoi": {
            "center_lon": coords[0][0],
            "center_lat": coords[0][1],
            "bounds": coords,
        },
        "land_cover_statistics": {k: v for k, v in stats.items() if not k.startswith("_")},
    }
    metadata_path = cfg.output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path


def run_landcover_pipeline(aoi_cfg: AOIConfig, dl_cfg: DownloadConfig) -> Dict[str, Path]:
    """End-to-end workflow: download Sentinel-2, Dynamic World products, stats, viz, masks."""
    dl_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    init_ee(dl_cfg.ee_project)

    aoi = create_aoi(aoi_cfg)

    s2_collection = load_sentinel2(aoi, dl_cfg)
    if s2_collection.size().getInfo() == 0:
        raise RuntimeError("No Sentinel-2 scenes found for the requested window.")
    s2_composite = create_sentinel2_composite(s2_collection)

    dw_collection = load_dynamic_world(aoi, dl_cfg)
    if dw_collection.size().getInfo() == 0:
        raise RuntimeError("No Dynamic World scenes found for the requested window.")
    classification = create_classification_composite(dw_collection)
    probabilities = create_probability_composite(dw_collection)

    sentinel_path = download_geotiff(
        s2_composite, aoi, "sentinel2_hyperspectral.tif", dl_cfg
    )
    classification_path = download_geotiff(
        classification, aoi, "land_cover_classification.tif", dl_cfg, ["label"]
    )
    probability_path = download_geotiff(
        probabilities, aoi, "land_cover_probabilities.tif", dl_cfg
    )

    veg_mask = create_vegetation_mask(dw_collection, threshold=0.27)
    veg_mask_path = download_geotiff(
        veg_mask, aoi, "vegetation_mask.tif", dl_cfg
    )

    stats = calculate_statistics(classification_path, dl_cfg)
    viz_path = create_visualizations(
        classification_path, probability_path, sentinel_path, stats, dl_cfg
    )
    metadata_path = save_metadata(stats, aoi, dl_cfg)

    return {
        "sentinel2": sentinel_path,
        "classification": classification_path,
        "probabilities": probability_path,
        "visualization": viz_path,
        "vegetation_mask": veg_mask_path,   # <- added this
        "metadata": metadata_path,
    }


if __name__ == "__main__":
    # Set output_dir to the parent directory of this script
    parent_dir = Path(__file__).resolve().parent
    results_dir = parent_dir / "LandcoverResults"
    default_cfg = DownloadConfig(
        output_dir=results_dir,
        date_start="2024-10-01",
        date_end="2024-11-15",
        scale=10,
        cloud_cover_max=20,
    )
    default_aoi = AOIConfig(lon=77.303778, lat=28.560278, buffer_km=3.0)
    outputs = run_landcover_pipeline(default_aoi, default_cfg)
    for name, path in outputs.items():
        print(f"{name}: {path}")