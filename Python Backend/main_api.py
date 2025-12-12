# main_api.py -- Rewritten AgniVed API (landcover, vegetation, video)
from pathlib import Path
from typing import Optional, Dict, Any
from importlib.util import spec_from_file_location, module_from_spec
import sys
import logging
import inspect

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator

import uvicorn

# keep video import as-is (left untouched)
from Video_inference_engine.Video_inference import run_youtube_live_inference

# -----------------------------------------------------------------------------
# App + logging
# -----------------------------------------------------------------------------
app = FastAPI(title="AgniVed Backend API (refactor)", version="1.0.0")
logger = logging.getLogger("main_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_RES_DW_DIR = PROJECT_ROOT / "Final_Res_DW"
MIN_BUFFER_KM = 0.6  # reject AOIs smaller than this

# -----------------------------------------------------------------------------
# Helpers: dynamic module loader
# -----------------------------------------------------------------------------
def load_module_from_path(name: str, file_path: Path):
    spec = spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {file_path}")
    module = module_from_spec(spec)
    # register module before exec so dataclasses and imports behave
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load landcover module eagerly (we rely on this)
landcover_path = (
    Path(__file__).parent
    / "S2 Landcover pipeline"
    / "Download_Classify.py"
)
landcover_mod = load_module_from_path("s2_landcover", landcover_path)
DW_AOIConfig = landcover_mod.AOIConfig
DownloadConfig = landcover_mod.DownloadConfig
run_landcover_pipeline = landcover_mod.run_landcover_pipeline

# Vegetation will be lazy-loaded to avoid heavy import during startup
VEG_AOIConfig = None
run_bigearth_rdnet = None
vegetation_mod_path = Path(__file__).parent / "S2 Vegetation Classification pipeline" / "Vegetation_Classification_pipeline.py"

def ensure_vegetation_module_loaded():
    global VEG_AOIConfig, run_bigearth_rdnet
    if VEG_AOIConfig is not None and run_bigearth_rdnet is not None:
        return
    vegetation_mod = load_module_from_path("s2_vegetation", vegetation_mod_path)
    VEG_AOIConfig = vegetation_mod.AOIConfig
    run_bigearth_rdnet = vegetation_mod.run_bigearth_rdnet
    logger.info(f"Vegetation module loaded from: {getattr(vegetation_mod, '__file__', 'unknown')}")

# -----------------------------------------------------------------------------
# Request models (km-only)
# -----------------------------------------------------------------------------
class LandcoverRequest(BaseModel):
    lon: float
    lat: float
    buffer_km: float = Field(MIN_BUFFER_KM, description="AOI radius in kilometers (>= 0.6)")
    date_start: str = "2024-10-01"
    date_end: str = "2024-11-15"
    scale: int = 10
    cloud_cover_max: int = 20

    @validator("buffer_km")
    def min_buffer(cls, v):
        if v < MIN_BUFFER_KM:
            raise ValueError(f"buffer_km must be >= {MIN_BUFFER_KM}")
        return v

class VegetationRequest(BaseModel):
    lon: float
    lat: float
    buffer_km: float = Field(MIN_BUFFER_KM, description="AOI radius in kilometers (>= 0.6)")
    use_mask: bool = True

    @validator("buffer_km")
    def min_buffer(cls, v):
        if v < MIN_BUFFER_KM:
            raise ValueError(f"buffer_km must be >= {MIN_BUFFER_KM}")
        return v

class VideoRequest(BaseModel):
    youtube_url: str

class PipelineRequest(BaseModel):
    lon: float
    lat: float
    buffer_km: float = Field(MIN_BUFFER_KM, description="DW AOI radius km (>= 0.6)")
    veg_buffer_km: float = Field(MIN_BUFFER_KM, description="Vegetation AOI radius km (>= 0.6)")
    date_start: str = "2024-10-01"
    date_end: str = "2024-11-15"
    scale: int = 10
    cloud_cover_max: int = 20

    @validator("buffer_km", "veg_buffer_km")
    def min_buffer(cls, v):
        if v < MIN_BUFFER_KM:
            raise ValueError(f"buffer must be >= {MIN_BUFFER_KM} km")
        return v

# -----------------------------------------------------------------------------
# Utility: safe file response
# -----------------------------------------------------------------------------
def serve_project_file(path: Path) -> FileResponse:
    p = path
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p = p.resolve()
    if not str(p).startswith(str(PROJECT_ROOT)):
        raise HTTPException(status_code=400, detail="Path outside project root not allowed.")
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {p}")
    return FileResponse(p)

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/landcover/dw")
def landcover_dw(req: LandcoverRequest):
    logger.info(f"landcover/dw called: lon={req.lon}, lat={req.lat}, buffer_km={req.buffer_km}")
    aoi = DW_AOIConfig(lon=req.lon, lat=req.lat, buffer_km=req.buffer_km)
    cfg = DownloadConfig(
        output_dir=FINAL_RES_DW_DIR,
        date_start=req.date_start,
        date_end=req.date_end,
        scale=req.scale,
        cloud_cover_max=req.cloud_cover_max,
    )

    try:
        outputs = run_landcover_pipeline(aoi, cfg)
    except Exception as e:
        logger.exception("Landcover pipeline failed")
        raise HTTPException(status_code=500, detail=f"Landcover pipeline failed: {e}")

    # Make sure the vegetation mask is included (some older versions omitted it)
    serialized = {k: str(v) for k, v in outputs.items()}
    logger.info("Landcover pipeline completed. Outputs: %s", list(serialized.keys()))
    return {"status": "ok", "aoi": req.dict(), "outputs": serialized}

@app.post("/vegetation/bigearth")
def vegetation_bigearth(req: VegetationRequest):
    logger.info(f"vegetation/bigearth called: lon={req.lon}, lat={req.lat}, buffer_km={req.buffer_km}, use_mask={req.use_mask}")
    ensure_vegetation_module_loaded()

    veg_mask_path: Optional[Path] = None
    if req.use_mask:
        veg_mask_path = FINAL_RES_DW_DIR / "vegetation_mask.tif"
        if not veg_mask_path.exists():
            raise HTTPException(status_code=400, detail=f"vegetation_mask.tif not found at {veg_mask_path}. Run /landcover/dw first.")

    # Build AOI in km units (VEG_AOIConfig expects buffer_km per your change)
    aoi = VEG_AOIConfig(lon=req.lon, lat=req.lat, buffer_km=req.buffer_km)
    logger.info(f"Created vegetation AOI config with buffer {aoi.buffer_km} km")

    try:
        res = run_bigearth_rdnet(aoi, veg_mask_path=veg_mask_path)
    except Exception as e:
        logger.exception("BigEarth pipeline failed")
        raise HTTPException(status_code=500, detail=f"BigEarth pipeline failed: {e}")

    return {
        "status": "ok",
        "aoi": req.dict(),
        "cube_path": str(res.cube_path),
        "viz_path": str(res.viz_path),
        "class_distribution": res.class_distribution,
        "tile_counts": res.tile_counts,
        "avg_confidence": res.avg_confidence,
        "tiles_shape": res.tiles_shape,
    }

@app.post("/video/classify")
def video_classify(req: VideoRequest, background_tasks: BackgroundTasks):
    logger.info(f"video/classify called: {req.youtube_url}")
    # start background inference (unchanged behavior)
    background_tasks.add_task(run_youtube_live_inference, req.youtube_url)
    return {
        "status": "started",
        "youtube_url": req.youtube_url,
        "note": "Inference running in background; a window will open on the server.",
    }

@app.get("/files/image")
def get_image(path: str = Query(..., description="Absolute or project-relative path")):
    p = Path(path)
    return serve_project_file(p)

@app.post("/pipeline/run")
def pipeline_run(req: PipelineRequest):
    logger.info(f"pipeline/run called: lon={req.lon}, lat={req.lat}, buffer_km={req.buffer_km}, veg_buffer_km={req.veg_buffer_km}")
    # 1) Run landcover DW
    aoi_dw = DW_AOIConfig(lon=req.lon, lat=req.lat, buffer_km=req.buffer_km)
    cfg = DownloadConfig(
        output_dir=FINAL_RES_DW_DIR,
        date_start=req.date_start,
        date_end=req.date_end,
        scale=req.scale,
        cloud_cover_max=req.cloud_cover_max,
    )
    try:
        dw_outputs = run_landcover_pipeline(aoi_dw, cfg)
    except Exception as e:
        logger.exception("Landcover pipeline failed inside pipeline/run")
        raise HTTPException(status_code=500, detail=f"Landcover pipeline failed: {e}")

    # Strict: require vegetation_mask in outputs
    veg_mask_path = dw_outputs.get("vegetation_mask")
    if veg_mask_path is None:
        logger.error("Landcover outputs did not include vegetation_mask")
        raise HTTPException(status_code=500, detail="Landcover pipeline did not return a 'vegetation_mask' output.")

    veg_mask_path = Path(veg_mask_path)
    if not veg_mask_path.exists():
        logger.error("vegetation_mask file not found at %s", veg_mask_path)
        raise HTTPException(status_code=500, detail=f"Vegetation mask file not found at: {veg_mask_path}")

    # 2) Run vegetation using that mask
    ensure_vegetation_module_loaded()
    aoi_veg = VEG_AOIConfig(lon=req.lon, lat=req.lat, buffer_km=req.veg_buffer_km)
    logger.info(f"Created vegetation AOI config with buffer {aoi_veg.buffer_km} km")
    try:
        veg_res = run_bigearth_rdnet(aoi_veg, veg_mask_path=veg_mask_path)
    except Exception as e:
        logger.exception("BigEarth pipeline failed inside pipeline/run")
        raise HTTPException(status_code=500, detail=f"BigEarth pipeline failed: {e}")

    return {
        "status": "ok",
        "aoi": req.dict(),
        "landcover": {k: str(v) for k, v in dw_outputs.items()},
        "vegetation": {
            "cube_path": str(veg_res.cube_path),
            "viz_path": str(veg_res.viz_path),
            "class_distribution": veg_res.class_distribution,
            "tile_counts": veg_res.tile_counts,
            "avg_confidence": veg_res.avg_confidence,
            "tiles_shape": veg_res.tiles_shape,
        },
    }

# -----------------------------------------------------------------------------
# Debug: return loaded module file paths / AOI signatures
# -----------------------------------------------------------------------------
@app.get("/_debug/status")
def debug_status():
    ensure_vegetation_module_loaded()
    status = {
        "project_root": str(PROJECT_ROOT),
        "landcover_module": getattr(landcover_mod, "__file__", None),
        "vegetation_module": getattr(sys.modules.get("s2_vegetation"), "__file__", None)
                          if sys.modules.get("s2_vegetation") else None,
        "veg_aoi_params": list(inspect.signature(VEG_AOIConfig).parameters.keys())
                          if VEG_AOIConfig is not None else None,
        "dw_aoi_params": list(inspect.signature(DW_AOIConfig).parameters.keys())
                         if DW_AOIConfig is not None else None,
    }
    return status

# -----------------------------------------------------------------------------
# Run app
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000, reload=True)
