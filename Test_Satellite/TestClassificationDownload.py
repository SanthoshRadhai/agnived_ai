"""
Dynamic World + Sentinel-2 Hyperspectral Downloader
Downloads land cover classification AND raw hyperspectral imagery
"""

import sys
if sys.version_info >= (3, 0):
    import io
    sys.modules['StringIO'] = io

import ee
import geemap
from pathlib import Path
from datetime import datetime
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import json

# Initialize Earth Engine
try:
    ee.Initialize(project='our-lamp-465108-a9')
    print("✓ Earth Engine initialized")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Configuration
CONFIG = {
    'output_dir': Path("../Test_results_DW/"),
    'date_start': '2024-10-01',
    'date_end': '2024-11-15',
    'scale': 10,
    'cloud_cover_max': 20,
    'aoi': {
        'center_lon': 77.303778,
        'center_lat': 28.560278,
        'buffer_km': 3.0
    }
}

# Dynamic World Land Cover Classes
DYNAMIC_WORLD_CLASSES = {
    0: {'name': 'Water', 'color': '#419BDF', 'description': 'Permanent and seasonal water bodies'},
    1: {'name': 'Trees', 'color': '#397D49', 'description': 'Forests, tree plantations, orchards'},
    2: {'name': 'Grass', 'color': '#88B053', 'description': 'Natural grasslands, pastures, parks'},
    3: {'name': 'Flooded Vegetation', 'color': '#7A87C6', 'description': 'Mangroves, wetlands'},
    4: {'name': 'Crops', 'color': '#E49635', 'description': 'Agricultural land, croplands'},
    5: {'name': 'Shrub & Scrub', 'color': '#DFC35A', 'description': 'Shrublands, scrublands, bushes'},
    6: {'name': 'Built Area', 'color': '#C4281B', 'description': 'Urban, buildings, roads'},
    7: {'name': 'Bare Ground', 'color': '#A59B8F', 'description': 'Exposed soil, sand, rocks'},
    8: {'name': 'Snow & Ice', 'color': '#B39FE1', 'description': 'Snow cover, glaciers'}
}

# Sentinel-2 Bands
SENTINEL2_BANDS = {
    'B1': {'name': 'Coastal Aerosol', 'wavelength': '443nm', 'resolution': 60},
    'B2': {'name': 'Blue', 'wavelength': '490nm', 'resolution': 10},
    'B3': {'name': 'Green', 'wavelength': '560nm', 'resolution': 10},
    'B4': {'name': 'Red', 'wavelength': '665nm', 'resolution': 10},
    'B5': {'name': 'Red Edge 1', 'wavelength': '705nm', 'resolution': 20},
    'B6': {'name': 'Red Edge 2', 'wavelength': '740nm', 'resolution': 20},
    'B7': {'name': 'Red Edge 3', 'wavelength': '783nm', 'resolution': 20},
    'B8': {'name': 'NIR', 'wavelength': '842nm', 'resolution': 10},
    'B8A': {'name': 'NIR Narrow', 'wavelength': '865nm', 'resolution': 20},
    'B9': {'name': 'Water Vapor', 'wavelength': '945nm', 'resolution': 60},
    'B11': {'name': 'SWIR 1', 'wavelength': '1610nm', 'resolution': 20},
    'B12': {'name': 'SWIR 2', 'wavelength': '2190nm', 'resolution': 20}
}

# AgniVed 7-class mapping
AGNIVED_MAPPING = {
    'Dense Forest': [1],
    'Evergreen': [1],
    'Deciduous': [1],
    'Grassland': [2],
    'Shrubland': [5],
    'Agricultural': [4],
    'Barren': [7]
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

def create_aoi():
    """Create area of interest"""
    center = ee.Geometry.Point([CONFIG['aoi']['center_lon'], CONFIG['aoi']['center_lat']])
    buffer_meters = CONFIG['aoi']['buffer_km'] * 1000
    aoi = center.buffer(buffer_meters).bounds()
    
    coords = aoi.coordinates().getInfo()[0]
    print(f"\n✓ AOI Created")
    print(f"   Center: ({CONFIG['aoi']['center_lat']:.4f}, {CONFIG['aoi']['center_lon']:.4f})")
    print(f"   Size: ~{CONFIG['aoi']['buffer_km']*2:.1f}km × {CONFIG['aoi']['buffer_km']*2:.1f}km")
    print(f"   Bounds: [{coords[0][0]:.4f}, {coords[0][1]:.4f}] to [{coords[2][0]:.4f}, {coords[2][1]:.4f}]")
    
    return aoi

def load_sentinel2(aoi):
    """Load Sentinel-2 hyperspectral imagery"""
    print("\n📥 Loading Sentinel-2 data...")
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(CONFIG['date_start'], CONFIG['date_end'])
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG['cloud_cover_max']))
    )
    
    count = collection.size().getInfo()
    print(f"✓ Found {count} Sentinel-2 images (cloud cover < {CONFIG['cloud_cover_max']}%)")
    
    if count > 0:
        first_image = collection.first()
        date = ee.Date(first_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        cloud_cover = first_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        print(f"   First image date: {date}")
        print(f"   Cloud cover: {cloud_cover:.1f}%")
        
        bands = first_image.bandNames().getInfo()
        print(f"   Total bands available: {len(bands)}")
    
    return collection

def create_sentinel2_composite(collection):
    """Create median composite of Sentinel-2"""
    print("\n🛰️  Creating Sentinel-2 composite (median, cloud-free)...")
    
    # Select all spectral bands
    spectral_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    
    composite = collection.select(spectral_bands).median()
    
    print("✓ Sentinel-2 composite created")
    print(f"   Bands: {spectral_bands}")
    print(f"   12 spectral bands (Coastal Aerosol → SWIR2)")
    
    return composite

def load_dynamic_world(aoi):
    """Load Dynamic World collection"""
    print("\n📥 Loading Dynamic World data...")
    
    collection = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        .filterBounds(aoi)
        .filterDate(CONFIG['date_start'], CONFIG['date_end'])
    )
    
    count = collection.size().getInfo()
    print(f"✓ Found {count} Dynamic World images")
    
    if count > 0:
        first_image = collection.first()
        date = ee.Date(first_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        print(f"   First image date: {date}")
        print(f"   Classification band: label (0-8 class IDs)")
    
    return collection

def create_classification_composite(collection):
    """Create mode composite (most frequent class)"""
    print("\n🗺️  Creating classification composite (mode)...")
    
    labels = collection.select('label')
    classification = labels.mode()
    
    print("✓ Classification composite created")
    return classification

def create_probability_composite(collection):
    """Create mean probability composite for all classes"""
    print("\n📊 Creating probability composites...")
    
    probability_bands = ['water', 'trees', 'grass', 'flooded_vegetation', 
                        'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
    
    probabilities = collection.select(probability_bands).mean()
    
    print("✓ Probability composites created")
    return probabilities

def calculate_statistics(classification, aoi):
    """Calculate land cover statistics"""
    print("\n📈 Calculating land cover statistics...")
    
    class_areas = classification.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=aoi,
        scale=CONFIG['scale'],
        maxPixels=1e13
    ).getInfo()
    
    histogram = class_areas.get('label', {})
    if isinstance(histogram, str):
        histogram = json.loads(histogram)
    
    total_pixels = sum(histogram.values())
    pixel_area_m2 = CONFIG['scale'] * CONFIG['scale']
    
    stats = {}
    for class_id_str, pixel_count in histogram.items():
        class_id = int(float(class_id_str))
        if class_id in DYNAMIC_WORLD_CLASSES:
            class_info = DYNAMIC_WORLD_CLASSES[class_id]
            area_km2 = (pixel_count * pixel_area_m2) / 1e6
            percentage = (pixel_count / total_pixels) * 100
            
            stats[class_info['name']] = {
                'pixels': pixel_count,
                'area_km2': area_km2,
                'percentage': percentage,
                'description': class_info['description']
            }
    
    print("✓ Statistics calculated")
    return stats

def download_geotiff(image, aoi, filename, bands=None):
    """Download image as GeoTIFF"""
    output_path = CONFIG['output_dir'] / filename
    
    print(f"\n📥 Downloading: {filename}")
    
    try:
        if bands:
            image = image.select(bands)
        
        geemap.download_ee_image(
            image,
            filename=str(output_path),
            scale=CONFIG['scale'],
            region=aoi,
            crs='EPSG:4326'
        )
        
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ Saved: {output_path.name} ({file_size_mb:.2f} MB)")
            return output_path
        else:
            print(f"❌ File not created")
            return None
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def create_visualizations(classification_path, probabilities_path, sentinel2_path, stats):
    """Create comprehensive visualizations"""
    print("\n🎨 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Dynamic World + Sentinel-2 Analysis', fontsize=18, fontweight='bold')
    
    # Load classification
    with rasterio.open(classification_path) as src:
        classification_data = src.read(1)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    
    # Load Sentinel-2
    with rasterio.open(sentinel2_path) as src:
        # RGB composite (B4=Red, B3=Green, B2=Blue)
        red = src.read(4)
        green = src.read(3)
        blue = src.read(2)
        
        # Normalize to 0-1
        rgb = np.dstack([
            np.clip(red / 3000, 0, 1),
            np.clip(green / 3000, 0, 1),
            np.clip(blue / 3000, 0, 1)
        ])
    
    # Plot 1: Sentinel-2 True Color
    ax1 = axes[0, 0]
    ax1.imshow(rgb, extent=extent)
    ax1.set_title('Sentinel-2 True Color (RGB)', fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # Plot 2: Classification Map
    ax2 = axes[0, 1]
    colors = [DYNAMIC_WORLD_CLASSES[i]['color'] for i in range(9)]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    im2 = ax2.imshow(classification_data, cmap=cmap, vmin=0, vmax=8, extent=extent)
    ax2.set_title('Land Cover Classification', fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    
    patches = [mpatches.Patch(color=DYNAMIC_WORLD_CLASSES[i]['color'], 
                             label=DYNAMIC_WORLD_CLASSES[i]['name']) 
              for i in range(9) if i in DYNAMIC_WORLD_CLASSES]
    ax2.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1), fontsize=7)
    
    # Plot 3: Statistics Bar Chart
    ax3 = axes[0, 2]
    class_names = list(stats.keys())
    percentages = [stats[name]['percentage'] for name in class_names]
    colors_bar = [DYNAMIC_WORLD_CLASSES[i]['color'] for i, info in DYNAMIC_WORLD_CLASSES.items() 
                  if info['name'] in class_names]
    
    bars = ax3.barh(class_names, percentages, color=colors_bar)
    ax3.set_xlabel('Coverage (%)', fontweight='bold')
    ax3.set_title('Land Cover Distribution', fontweight='bold')
    ax3.set_xlim(0, 100)
    
    for i, (name, bar) in enumerate(zip(class_names, bars)):
        width = bar.get_width()
        ax3.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', va='center', fontsize=8)
    
    # Plot 4: Sentinel-2 False Color (NIR-Red-Green for vegetation)
    ax4 = axes[1, 0]
    with rasterio.open(sentinel2_path) as src:
        nir = src.read(8)
        
    false_color = np.dstack([
        np.clip(nir / 3000, 0, 1),
        np.clip(red / 3000, 0, 1),
        np.clip(green / 3000, 0, 1)
    ])
    
    ax4.imshow(false_color, extent=extent)
    ax4.set_title('Sentinel-2 False Color (NIR-R-G)\nVegetation appears red', fontweight='bold')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    
    # Plot 5: Tree Probability
    ax5 = axes[1, 1]
    with rasterio.open(probabilities_path) as src:
        trees_prob = src.read(2)
        
    im5 = ax5.imshow(trees_prob, cmap='Greens', vmin=0, vmax=1, extent=extent)
    ax5.set_title('Tree Coverage Probability', fontweight='bold')
    ax5.set_xlabel('Longitude')
    ax5.set_ylabel('Latitude')
    plt.colorbar(im5, ax=ax5, label='Probability')
    
    # Plot 6: Built Area Probability
    ax6 = axes[1, 2]
    with rasterio.open(probabilities_path) as src:
        built_prob = src.read(7)
        
    im6 = ax6.imshow(built_prob, cmap='Reds', vmin=0, vmax=1, extent=extent)
    ax6.set_title('Built Area Probability', fontweight='bold')
    ax6.set_xlabel('Longitude')
    ax6.set_ylabel('Latitude')
    plt.colorbar(im6, ax=ax6, label='Probability')
    
    plt.tight_layout()
    
    viz_path = CONFIG['output_dir'] / 'comprehensive_analysis.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {viz_path.name}")
    plt.close()
    
    # Create individual class masks
    print("\n🎭 Creating individual class masks...")
    
    for class_id, class_info in DYNAMIC_WORLD_CLASSES.items():
        mask = (classification_data == class_id).astype(np.uint8)
        
        if mask.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(mask, cmap='gray', extent=extent)
            ax.set_title(f"{class_info['name']} Mask", fontweight='bold', fontsize=14)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            
            mask_path = CONFIG['output_dir'] / f"mask_{class_info['name'].lower().replace(' ', '_')}.png"
            plt.savefig(mask_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"   ✓ {class_info['name']} mask saved")

def save_metadata(stats, aoi):
    """Save comprehensive metadata JSON"""
    print("\n💾 Saving metadata...")
    
    coords = aoi.coordinates().getInfo()[0]
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'data_sources': {
            'land_cover': 'Google Dynamic World V1',
            'hyperspectral': 'Sentinel-2 SR Harmonized'
        },
        'resolution': f"{CONFIG['scale']}m",
        'date_range': {
            'start': CONFIG['date_start'],
            'end': CONFIG['date_end']
        },
        'area_of_interest': {
            'center': {
                'latitude': CONFIG['aoi']['center_lat'],
                'longitude': CONFIG['aoi']['center_lon']
            },
            'bounds': {
                'west': coords[0][0],
                'south': coords[0][1],
                'east': coords[2][0],
                'north': coords[2][1]
            },
            'buffer_km': CONFIG['aoi']['buffer_km']
        },
        'sentinel2_bands': SENTINEL2_BANDS,
        'land_cover_statistics': stats,
        'agnived_mapping': {
            'description': 'Mapping from Dynamic World to AgniVed 7-class system',
            'classes': AGNIVED_MAPPING
        },
        'dynamic_world_classes': DYNAMIC_WORLD_CLASSES
    }
    
    metadata_path = CONFIG['output_dir'] / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved: {metadata_path.name}")

def print_summary(stats):
    """Print comprehensive summary"""
    print("\n" + "=" * 70)
    print("LAND COVER SUMMARY")
    print("=" * 70)
    
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['area_km2'], reverse=True)
    
    print(f"\n{'Class':<20} {'Area (km²)':<12} {'Coverage %':<12} {'Description'}")
    print("-" * 70)
    
    for class_name, data in sorted_stats:
        print(f"{class_name:<20} {data['area_km2']:>10.3f}  {data['percentage']:>10.1f}%  {data['description']}")
    
    total_area = sum(s['area_km2'] for s in stats.values())
    print("-" * 70)
    print(f"{'TOTAL':<20} {total_area:>10.3f} km²")

def main():
    print("=" * 70)
    print("DYNAMIC WORLD + SENTINEL-2 DOWNLOADER")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    aoi = create_aoi()
    
    # Load Sentinel-2
    s2_collection = load_sentinel2(aoi)
    
    if s2_collection.size().getInfo() == 0:
        print("\n❌ No Sentinel-2 data available")
        return
    
    # Create Sentinel-2 composite
    s2_composite = create_sentinel2_composite(s2_collection)
    
    # Load Dynamic World
    dw_collection = load_dynamic_world(aoi)
    
    if dw_collection.size().getInfo() == 0:
        print("\n❌ No Dynamic World data available")
        return
    
    # Create composites
    classification = create_classification_composite(dw_collection)
    probabilities = create_probability_composite(dw_collection)
    
    # Calculate statistics
    stats = calculate_statistics(classification, aoi)
    
    # Download GeoTIFFs
    print("\n" + "=" * 70)
    print("DOWNLOADING DATA")
    print("=" * 70)
    
    sentinel2_path = download_geotiff(s2_composite, aoi, "sentinel2_hyperspectral.tif")
    classification_path = download_geotiff(classification, aoi, "land_cover_classification.tif", ['label'])
    probabilities_path = download_geotiff(probabilities, aoi, "land_cover_probabilities.tif")
    
    if classification_path and probabilities_path and sentinel2_path:
        create_visualizations(classification_path, probabilities_path, sentinel2_path, stats)
        save_metadata(stats, aoi)
        print_summary(stats)
        
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n📁 Output Directory: {CONFIG['output_dir']}")
        print("\n📄 Generated Files:")
        print("   1. sentinel2_hyperspectral.tif - 12-band hyperspectral imagery")
        print("   2. land_cover_classification.tif - Classification map (0-8)")
        print("   3. land_cover_probabilities.tif - Probability bands")
        print("   4. comprehensive_analysis.png - 6-panel visualization")
        print("   5. mask_*.png - Individual class masks")
        print("   6. metadata.json - Complete metadata")
        print("\n🎯 Next: Use vegetation masks to filter Sentinel-2 for Prithvi input")

if __name__ == "__main__":
    main()