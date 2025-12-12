import ee
import geemap
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
# UPDATED IMPORTS FOR MOVIEPY 2.0+
from moviepy import ImageClip, ImageSequenceClip, clips_array
from pathlib import Path
from datetime import datetime, timedelta
import os

# --- 1. CONFIG & INIT ---
VEG_BANDS = ['trees', 'grass', 'crops', 'shrub_and_scrub']
WATER_BANDS = ['water', 'flooded_vegetation']

class ChangeResult:
    def __init__(self, percent_change, video_path, veg_heatmap_path, water_heatmap_path):
        self.percent_change = percent_change
        self.video_path = video_path
        self.veg_heatmap_path = veg_heatmap_path
        self.water_heatmap_path = water_heatmap_path

def init_ee(project_id="our-lamp-465108-a9"):
    try:
        ee.Initialize(project=project_id)
    except:
        ee.Authenticate()
        ee.Initialize(project=project_id)

# --- 2. HEATMAP GENERATION (Quantitative) ---
def generate_colored_heatmap(tif_path, out_png_path, mode='veg'):
    """Converts a raw difference GeoTIFF into a beautiful matplotlib PNG"""
    with rasterio.open(tif_path) as src:
        data = src.read(1)
    
    # Figure setup (Square for stacking)
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    if mode == 'veg':
        # Red (Loss) -> White (Stable) -> Green (Gain)
        cmap = mcolors.LinearSegmentedColormap.from_list("veg", ["#8B0000", "#FFFFFF", "#006400"])
        title = "VEGETATION CHANGE"
    else:
        # Brown (Loss) -> White (Stable) -> Blue (Gain)
        cmap = mcolors.LinearSegmentedColormap.from_list("water", ["#8B4513", "#FFFFFF", "#00008B"])
        title = "WATER CHANGE"

    # Clamp values to -0.5 to 0.5 for best visual contrast
    plt.imshow(data, cmap=cmap, vmin=-0.5, vmax=0.5, aspect='auto')
    
    # Add title
    plt.text(0.5, 0.95, title, horizontalalignment='center', verticalalignment='top', 
             transform=ax.transAxes, color='black', fontsize=14, weight='bold', 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.savefig(out_png_path)
    plt.close()
    return str(out_png_path)

def get_probability_diff(aoi, start_date, end_date):
    """Calculates Bi-Temporal Difference (End - Start)"""
    s_date = datetime.strptime(start_date, "%Y-%m-%d")
    e_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Define 30-day windows
    p1_start = s_date.strftime("%Y-%m-%d")
    p1_end = (s_date + timedelta(days=30)).strftime("%Y-%m-%d")
    p2_start = (e_date - timedelta(days=30)).strftime("%Y-%m-%d")
    p2_end = e_date.strftime("%Y-%m-%d")

    dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(aoi)
    
    img_t1 = dw.filterDate(p1_start, p1_end).select(VEG_BANDS + WATER_BANDS).mean()
    img_t2 = dw.filterDate(p2_start, p2_end).select(VEG_BANDS + WATER_BANDS).mean()

    # Sum bands first, then subtract
    veg_t1 = img_t1.select(VEG_BANDS).reduce(ee.Reducer.sum())
    veg_t2 = img_t2.select(VEG_BANDS).reduce(ee.Reducer.sum())
    veg_diff = veg_t2.subtract(veg_t1)
    
    water_t1 = img_t1.select(WATER_BANDS).reduce(ee.Reducer.sum())
    water_t2 = img_t2.select(WATER_BANDS).reduce(ee.Reducer.sum())
    water_diff = water_t2.subtract(water_t1)
    
    # Stats
    stat = veg_diff.reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, scale=100, maxPixels=1e9).getInfo()
    net_change = stat.get('sum', 0) * 100 

    return veg_diff, water_diff, net_change

# --- 3. TIMELAPSE GENERATION ---
def download_s2_frames(aoi, start_date, end_date, output_dir, num_frames=12):
    s = datetime.strptime(start_date, "%Y-%m-%d")
    e = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (e - s).days
    step = max(1, total_days // num_frames)
    
    frame_paths = []
    
    for i in range(num_frames):
        d1 = (s + timedelta(days=i*step)).strftime("%Y-%m-%d")
        d2 = (s + timedelta(days=(i+1)*step)).strftime("%Y-%m-%d")
        
        # Get median composite
        s2_img = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
            .filterBounds(aoi).filterDate(d1, d2)\
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))\
            .median()\
            .visualize(min=0, max=3000, bands=['B4', 'B3', 'B2'])
        
        fname = output_dir / f"frame_{i:02d}.tif"
        png_name = output_dir / f"frame_{i:02d}.png"
        
        try:
            geemap.download_ee_image(s2_img, filename=str(fname), scale=10, region=aoi, crs='EPSG:4326')
            # Convert TIF to PNG for moviepy
            with rasterio.open(fname) as src:
                arr = src.read()
                # [B,H,W] -> [H,W,B]
                arr = np.transpose(arr, (1, 2, 0))
                plt.imsave(str(png_name), arr)
                frame_paths.append(str(png_name))
        except:
            pass 
            
    return frame_paths

# --- 4. MAIN PIPELINE ---
def ChangeDetection_HM(lat, lon, radius_km, start_date, end_date, output_dir):
    init_ee()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    point = ee.Geometry.Point([lon, lat])
    aoi = point.buffer(radius_km * 1000).bounds()

    print("1. Generating Heatmaps...")
    veg_diff, water_diff, net_change = get_probability_diff(aoi, start_date, end_date)
    
    veg_tif = output_dir / "veg_diff.tif"
    water_tif = output_dir / "water_diff.tif"
    
    geemap.download_ee_image(veg_diff, filename=str(veg_tif), scale=10, region=aoi, crs='EPSG:4326')
    geemap.download_ee_image(water_diff, filename=str(water_tif), scale=10, region=aoi, crs='EPSG:4326')
    
    veg_png_path = output_dir / "veg_heatmap.png"
    water_png_path = output_dir / "water_heatmap.png"
    
    generate_colored_heatmap(veg_tif, veg_png_path, 'veg')
    generate_colored_heatmap(water_tif, water_png_path, 'water')

    print("2. Downloading Timelapse Frames...")
    frames = download_s2_frames(aoi, start_date, end_date, output_dir)
    
    print("3. Composing Final Video...")
    if not frames:
        raise Exception("Could not generate timelapse frames (cloud cover too high?)")

    fps = 1
    duration = len(frames) / fps
    
    # Create Clips using MoviePy v2 compatible resizing
    # Note: in v2, resize is a method of the clip object directly
    try:
        # Left Panel (Heatmaps stacked)
        clip_veg = ImageClip(str(veg_png_path)).with_duration(duration).resized(height=360)
        clip_water = ImageClip(str(water_png_path)).with_duration(duration).resized(height=360)
        left_panel = clips_array([[clip_veg], [clip_water]])
        
        # Right Panel (Timelapse)
        right_panel = ImageSequenceClip(frames, fps=fps).resized(height=720)
        
        # Final Composite
        final_video = clips_array([[left_panel, right_panel]])
        
        video_filename = "change_analysis_composite.mp4"
        video_path = output_dir / video_filename
        
        final_video.write_videofile(str(video_path), codec='libx264', fps=24, logger=None)
        
    except AttributeError:
        # Fallback for older MoviePy versions (uses .resize instead of .resized)
        clip_veg = ImageClip(str(veg_png_path)).set_duration(duration).resize(height=360)
        clip_water = ImageClip(str(water_png_path)).set_duration(duration).resize(height=360)
        left_panel = clips_array([[clip_veg], [clip_water]])
        right_panel = ImageSequenceClip(frames, fps=fps).resize(height=720)
        final_video = clips_array([[left_panel, right_panel]])
        
        video_filename = "change_analysis_composite.mp4"
        video_path = output_dir / video_filename
        final_video.write_videofile(str(video_path), codec='libx264', fps=24, logger=None)

    return ChangeResult(
        round(net_change, 2), 
        video_filename, 
        veg_heatmap_path="veg_heatmap.png", 
        water_heatmap_path="water_heatmap.png"
    )