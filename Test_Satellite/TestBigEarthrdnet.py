import sys
from pathlib import Path
import shutil
import requests
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import ee

# -----------------------------------------------------------------------------#
# Configuration                                                                #
# -----------------------------------------------------------------------------#
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

REBEN_DIR = SCRIPT_DIR / "reben"
TEMP_DIR = Path("/tmp/bigearth_rdnet_s2")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

AOI_CENTER = (-1.3972, 50.9069)   # (lon, lat)
AOI_BUFFER = 600                 # ~1.2 km -> ~120x120 px at 10 m

SEASONS = [
    ("2024-01-01", "2024-03-31", "Winter"),
    ("2024-04-01", "2024-06-30", "Spring"),
    ("2024-07-01", "2024-09-30", "Summer"),
    ("2024-10-01", "2024-11-15", "Fall"),
]

S2_BANDS = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
PATCH_SIZE = 120

BEN_CLASSES = [
    "Agro-forestry areas","Arable land","Beaches, dunes, sands","Broad-leaved forest",
    "Coastal wetlands","Complex cultivation patterns","Coniferous forest",
    "Industrial or commercial units","Inland waters","Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters","Mixed forest","Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas","Pastures","Permanent crops",
    "Transitional woodland, shrub","Urban fabric",
]

# Same DN-space stats you used for ConvNeXtS2 so comparison is fair
BEN_MEAN = np.array([
    1370.19151926,1184.38246250,1120.77120066,1136.26026392,1263.73947144,
    1645.40315151,1846.87040806,1762.59530783,1972.62420416,582.72633433,
])
BEN_STD = np.array([
    633.15169573,650.28427720,712.12507725,965.23119807,948.98199320,
    1108.06650639,1258.36394548,1233.14922810,1364.38688993,472.37967789,
])

# -----------------------------------------------------------------------------#
# Utilities                                                                    #
# -----------------------------------------------------------------------------#
def load_reben_model(device: torch.device) -> torch.nn.Module:
    if not REBEN_DIR.exists():
        raise SystemExit("❌ reben directory missing. Place it at Test_Satellite/reben.")
    sys.path.insert(0, str(REBEN_DIR))
    try:
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
    except ImportError as exc:
        raise SystemExit(f"❌ Failed to import reBEN modules: {exc}")
    # RdNet S2 model
    model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
        "BIFOLD-BigEarthNetv2-0/rdnet_base-s2-v0.2.0",
        map_location=("cuda:0" if device.type == "cuda" else "cpu"),
    )
    model = model.to(device)
    model.eval()
    return model


def init_earth_engine():
    try:
        ee.Initialize(project="our-lamp-465108-a9")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="our-lamp-465108-a9")


def seasonal_median(start, end, label, aoi):
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                  .select(S2_BANDS))
    count = collection.size().getInfo()
    if count == 0:
        raise RuntimeError(f"No Sentinel-2 scenes for {label}.")
    print(f"   {label}: {count} images")
    return collection.median().clip(aoi)


def download_single_composite(aoi):
    start = SEASONS[0][0]
    end = SEASONS[-1][1]
    print(f"\n🛰️  Building SINGLE median composite from {start} to {end}...")

    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                  .select(S2_BANDS))

    count = collection.size().getInfo()
    if count == 0:
        raise RuntimeError("No Sentinel-2 scenes for the requested period.")
    print(f"   Images in collection: {count}")

    composite = collection.median().clip(aoi)

    url = composite.getDownloadURL({
        "scale": 10,
        "region": aoi,
        "filePerBand": False,
        "format": "GEO_TIFF",
    })
    out_tif = TEMP_DIR / "s2_single_stack.tif"
    with requests.get(url, stream=True) as resp, open(out_tif, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    with rasterio.open(out_tif) as src:
        data = src.read()
    return data


def tile_cube(cube):
    _, H, W = cube.shape
    h_tiles = H // PATCH_SIZE
    w_tiles = W // PATCH_SIZE
    if h_tiles == 0 or w_tiles == 0:
        raise RuntimeError("AOI too small to form a 120×120 patch.")
    tiles, coords = [], []
    for i in range(h_tiles):
        for j in range(w_tiles):
            y, x = i * PATCH_SIZE, j * PATCH_SIZE
            tiles.append(cube[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE])
            coords.append((i, j, y, x))
    return np.stack(tiles), coords, h_tiles, w_tiles


def normalize_tiles(tiles):
    return (tiles - BEN_MEAN[:, None, None]) / BEN_STD[:, None, None]


# -----------------------------------------------------------------------------#
# Main                                                                          #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    print("=" * 70)
    print("BIGEARTH RDNET-S2 – GEE PATCH TEST")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    init_earth_engine()
    point = ee.Geometry.Point(AOI_CENTER)
    aoi = point.buffer(AOI_BUFFER).bounds()

    s2_cube = download_single_composite(aoi)
    print(f"   Composite shape: {s2_cube.shape}, range: [{s2_cube.min():.1f}, {s2_cube.max():.1f}]")

    tiles, coords, H_tiles, W_tiles = tile_cube(s2_cube)
    tiles_norm = normalize_tiles(tiles)

    model = load_reben_model(device)
    with torch.no_grad():
        tensor = torch.from_numpy(tiles_norm).float().to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    top3_idx = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    top3_scores = np.take_along_axis(probs, top3_idx, axis=1)

    print("\n" + "=" * 70)
    print("TOP-3 PREDICTIONS PER PATCH (first 5)")
    print("=" * 70)
    for n, (i, j, y, x) in enumerate(coords[:5]):
        print(f"\nPatch {n+1}  Grid[{i},{j}]  Pixels[{y}:{y+PATCH_SIZE}, {x}:{x+PATCH_SIZE}]")
        for rank in range(3):
            cls = BEN_CLASSES[top3_idx[n, rank]]
            score = top3_scores[n, rank]
            print(f"   {rank+1}. {cls:<65} {score:.1%}")

    class_map = top3_idx[:, 0].reshape(H_tiles, W_tiles)
    conf_map = top3_scores[:, 0].reshape(H_tiles, W_tiles)

    rgb = np.stack([s2_cube[2], s2_cube[1], s2_cube[0]], axis=-1)
    rgb = np.clip(rgb / 3000.0, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 True Color")
    axes[0].axis("off")

    im_class = axes[1].imshow(class_map, cmap="tab20", vmin=0, vmax=18)
    axes[1].set_title("BigEarthNet Top-1 Class (RdNet S2)")
    axes[1].axis("off")
    legend_handles = [
        mpatches.Patch(color=plt.cm.tab20(c / 19), label=BEN_CLASSES[c])
        for c in np.unique(class_map)
    ]
    axes[1].legend(handles=legend_handles, bbox_to_anchor=(1.05, 1),
                   loc="upper left", fontsize=8)

    im_conf = axes[2].imshow(conf_map, cmap="RdYlGn", vmin=0, vmax=1)
    axes[2].set_title("Confidence")
    axes[2].axis("off")
    plt.colorbar(im_conf, ax=axes[2], fraction=0.046, pad=0.04)

    out_png = SCRIPT_DIR / "bigearth_rdnet_s2_results.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"\n✅ Saved visualization: {out_png}")

    counts = np.bincount(class_map.flatten(), minlength=len(BEN_CLASSES))
    print("\nCLASS DISTRIBUTION")
    for idx in counts.argsort()[::-1]:
        if counts[idx] == 0:
            continue
        pct = 100 * counts[idx] / counts.sum()
        print(f"  {BEN_CLASSES[idx]:<65} {pct:5.1f}% ({counts[idx]} tiles)")
    print(f"\nAverage confidence: {conf_map.mean():.1%}")

    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("\n Completed.")