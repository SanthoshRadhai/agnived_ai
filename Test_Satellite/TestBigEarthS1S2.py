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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REBEN_DIR = SCRIPT_DIR / "reben"
TEMP_DIR = Path("/tmp/bigearth_all")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# AOI in central Europe (you can change this)
AOI_CENTER = (-1.6003, 50.855)  # (lon, lat)
AOI_BUFFER = 600                 # ~1.2 km -> ~120x120 px at 10 m



SEASONS = [
    ("2024-01-01", "2024-03-31", "Winter"),
    ("2024-04-01", "2024-06-30", "Spring"),
    ("2024-07-01", "2024-09-30", "Summer"),
    ("2024-10-01", "2024-11-15", "Fall"),
]

# 10 S2 bands used in bandconfig "s2"
S2_BANDS = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
# 2 S1 bands (VV, VH) in IW GRD product
S1_BANDS = ["VV", "VH"]

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

# Simple reflectance stats for S2 (0–1 range)
S2_MEAN = np.array([0.10,0.12,0.12,0.20,0.23,0.24,0.25,0.27,0.18,0.12], dtype=np.float32)
S2_STD  = np.array([0.05,0.06,0.06,0.08,0.09,0.09,0.10,0.10,0.07,0.06], dtype=np.float32)

# Approx stats for log-scaled S1 backscatter (we'll log10 then std-normalize)
S1_MEAN = np.array([-11.0, -18.0], dtype=np.float32)  # VV, VH (dB-ish)
S1_STD  = np.array([  3.0,   3.0], dtype=np.float32)

def load_reben_model(device: torch.device) -> torch.nn.Module:
    if not REBEN_DIR.exists():
        raise SystemExit("❌ reben directory missing. Place it at Test_Satellite/reben.")
    sys.path.insert(0, str(REBEN_DIR))
    try:
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
    except ImportError as exc:
        raise SystemExit(f"❌ Failed to import reBEN modules: {exc}")
    model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
        "BIFOLD-BigEarthNetv2-0/convnextv2_base-all-v0.2.0",
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

def seasonal_median_s2(start, end, label, aoi):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi)
           .filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
           .select(S2_BANDS))
    count = col.size().getInfo()
    if count == 0:
        raise RuntimeError(f"No S2 scenes for {label}.")
    print(f"   S2 {label}: {count} images")
    return col.median().clip(aoi)

def seasonal_median_s1(start, end, label, aoi):
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(aoi)
           .filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
           .filter(ee.Filter.eq("resolution_meters", 10))
           .filter(ee.Filter.eq("polarization", "VVVH"))
           .select(S1_BANDS))
    count = col.size().getInfo()
    if count == 0:
        raise RuntimeError(f"No S1 scenes for {label}.")
    print(f"   S1 {label}: {count} images")
    return col.median().clip(aoi)

def download_composite_all(aoi):
    print("\n🛰️  Building seasonal composites (S2 + S1)...")
    s2_seasonals = [seasonal_median_s2(s,e,l,aoi) for (s,e,l) in SEASONS]
    s1_seasonals = [seasonal_median_s1(s,e,l,aoi) for (s,e,l) in SEASONS]

    s2_comp = ee.ImageCollection(s2_seasonals).median()
    s1_comp = ee.ImageCollection(s1_seasonals).median()

    # S2 download
    url_s2 = s2_comp.getDownloadURL({
        "scale": 10,
        "region": aoi,
        "filePerBand": False,
        "format": "GEO_TIFF",
    })
    out_s2 = TEMP_DIR / "s2_stack.tif"
    with requests.get(url_s2, stream=True) as resp, open(out_s2, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    with rasterio.open(out_s2) as src:
        s2_cube = src.read()  # (10,H,W)

    # S1 download
    url_s1 = s1_comp.getDownloadURL({
        "scale": 10,
        "region": aoi,
        "filePerBand": False,
        "format": "GEO_TIFF",
    })
    out_s1 = TEMP_DIR / "s1_stack.tif"
    with requests.get(url_s1, stream=True) as resp, open(out_s1, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    with rasterio.open(out_s1) as src:
        s1_cube = src.read()  # (2,H,W)

    # Make sure shapes match
    if s2_cube.shape[1:] != s1_cube.shape[1:]:
        H = min(s2_cube.shape[1], s1_cube.shape[1])
        W = min(s2_cube.shape[2], s1_cube.shape[2])
        s2_cube = s2_cube[:, :H, :W]
        s1_cube = s1_cube[:, :H, :W]

    # Stack into (12,H,W): [10 S2, 2 S1]
    all_cube = np.concatenate([s2_cube, s1_cube], axis=0)
    return all_cube, s2_cube  # return S2 separately for RGB plotting

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

def normalize_tiles_all(tiles):
    """
    tiles: (N,12,H,W) with channels=[10 S2 reflectances, 2 S1 power]
    """
    tiles = tiles.astype(np.float32)
    s2 = tiles[:, :10] / 10000.0                 # DN -> reflectance 0..1
    s2 = (s2 - S2_MEAN[None, :, None, None]) / S2_STD[None, :, None, None]

    s1 = tiles[:, 10:]                           # S1 backscatter
    s1 = np.clip(s1, 1e-6, None)
    s1_db = 10.0 * np.log10(s1)                  # to "dB"-ish
    s1 = (s1_db - S1_MEAN[None, :, None, None]) / S1_STD[None, :, None, None]

    return np.concatenate([s2, s1], axis=1)

def best_scene_s2(aoi):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi)
           .filterDate(SEASONS[0][0], SEASONS[-1][1])
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
           .select(S2_BANDS)
           .sort("CLOUDY_PIXEL_PERCENTAGE"))
    first = col.first()
    info = first.getInfo()
    print("   Using S2 scene:", info["id"],
          "clouds:", info["properties"]["CLOUDY_PIXEL_PERCENTAGE"])
    return first.clip(aoi)

def best_scene_s1(aoi):
    # Try descending orbit with VV/VH first, then relax filters if needed
    base = (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(SEASONS[0][0], SEASONS[-1][1])
            .filter(ee.Filter.eq("instrumentMode", "IW")))

    def pick(col, desc):
        col = col.sort("system:time_start")
        size = col.size().getInfo()
        if size == 0:
            print(f"   No S1 scenes ({desc}).")
            return None
        img = col.first()
        info = img.getInfo()
        print(f"   Using S1 scene ({desc}):", info["id"])
        return img.clip(aoi)

    # 1) IW, descending, VV/VH
    col1 = (base
            .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(S1_BANDS))
    img = pick(col1, "DESC VV+VH")
    if img is not None:
        return img

    # 2) IW, any orbit, VV/VH
    col2 = (base
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(S1_BANDS))
    img = pick(col2, "ANY VV+VH")
    if img is not None:
        return img

    # 3) Any IW scene (VV only, etc.)
    col3 = base.select(["VV"])
    img = pick(col3, "ANY VV-only")
    if img is not None:
        # duplicate VV as VV/VH placeholder
        img = img.addBands(img, overwrite=False).select(["VV", "VV"])
        return img.clip(aoi)

    raise RuntimeError("No Sentinel‑1 scenes in full period for this AOI.")


def download_best_scene_all(aoi):
    print("\n🛰️  Picking SINGLE best scenes (S2 + S1)...")
    s2_img = best_scene_s2(aoi)
    s1_img = best_scene_s1(aoi)

    # S2 download
    url_s2 = s2_img.getDownloadURL({
        "scale": 10,
        "region": aoi,
        "filePerBand": False,
        "format": "GEO_TIFF",
    })
    out_s2 = TEMP_DIR / "s2_stack.tif"
    with requests.get(url_s2, stream=True) as resp, open(out_s2, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    with rasterio.open(out_s2) as src:
        s2_cube = src.read()  # (10,H,W)

    # S1 download
    url_s1 = s1_img.getDownloadURL({
        "scale": 10,
        "region": aoi,
        "filePerBand": False,
        "format": "GEO_TIFF",
    })
    out_s1 = TEMP_DIR / "s1_stack.tif"
    with requests.get(url_s1, stream=True) as resp, open(out_s1, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    with rasterio.open(out_s1) as src:
        s1_cube = src.read()  # (2,H,W)

    # Align shapes if needed
    if s2_cube.shape[1:] != s1_cube.shape[1:]:
        H = min(s2_cube.shape[1], s1_cube.shape[1])
        W = min(s2_cube.shape[2], s1_cube.shape[2])
        s2_cube = s2_cube[:, :H, :W]
        s1_cube = s1_cube[:, :H, :W]

    all_cube = np.concatenate([s2_cube, s1_cube], axis=0)
    return all_cube, s2_cube


if __name__ == "__main__":
    print("=" * 70)
    print("BIGEARTH CONVNEXTS2-ALL – GEE PATCH TEST")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    init_earth_engine()
    point = ee.Geometry.Point(AOI_CENTER)
    aoi = point.buffer(AOI_BUFFER).bounds()

    # OLD:
    # all_cube, s2_cube = download_composite_all(aoi)
    # NEW:
    all_cube, s2_cube = download_best_scene_all(aoi)
    print(f"   All cube shape: {all_cube.shape}, "
          f"range: [{all_cube.min():.1f}, {all_cube.max():.1f}]")

    tiles, coords, H_tiles, W_tiles = tile_cube(all_cube)
    tiles_norm = normalize_tiles_all(tiles)

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

    # True color from S2 for visualization
    rgb = np.stack([s2_cube[2], s2_cube[1], s2_cube[0]], axis=-1)
    rgb = np.clip(rgb / 3000.0, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("Sentinel-2 True Color")
    axes[0].axis("off")

    im_class = axes[1].imshow(class_map, cmap="tab20", vmin=0, vmax=18)
    axes[1].set_title("BigEarthNet Top-1 Class (S1+S2)")
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

    out_png = SCRIPT_DIR / "bigearth_all_results.png"
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
    print("\n✅ Done.")