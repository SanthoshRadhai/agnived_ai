import glob
import os
import json
import traceback
import base64  # <--- Added for image encoding
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt

from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

from panos360Helper import *

# Import database utilities
from database import (
    init_db,
    get_connection,
    generate_uuid,
    get_uploads_in_radius,
    get_user_score,
    get_user_uploads
)

# Import the landcover pipeline
from landcover.LandCover import AOIConfig as LandcoverAOIConfig, DownloadConfig, run_landcover_pipeline
# Import the vegetation pipeline
from vegetation.Vegetation_Classification_pipeline import AOIConfig as VegAOIConfig, run_bigearth_rdnet

# GoogleVR4 imports
import GoogleVR4.core as core
from GoogleVR4.ObjectIdentifier import run_object_detection

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Plant model for classification
PLANT_MODEL_ID = "juppy44/plant-identification-2m-vit-b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plant_processor = AutoImageProcessor.from_pretrained(PLANT_MODEL_ID)
plant_model = AutoModelForImageClassification.from_pretrained(PLANT_MODEL_ID).to(DEVICE)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def image_to_base64(image_path: str) -> Optional[str]:
    """Reads an image file and converts it to a base64 string"""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

# -----------------------------------------------------------------------------
# Lifespan: Initialize DB on startup
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database
    init_db()
    print("✅ Database initialized and ready")
    yield
    # Shutdown: cleanup if needed
    pass

app = FastAPI(
    title="AgniVed Pipeline & Upload API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------

# Auth models
class UserRegister(BaseModel):
    userid: str
    name: str
    password: str
    role: str = "user"

class UserLogin(BaseModel):
    userid: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    userid: str

# Upload models
class UploadSearchRequest(BaseModel):
    latitude: float
    longitude: float
    radius_km: float = 10.0

# Pipeline models (existing)
class PanosRequest(BaseModel):
    lat: float
    lon: float
    count: int = 3
    area_of_interest: float = 100.0
    min_distance: float = 10.0

class DetectObjectsRequest(BaseModel):
    image_path: str
    labels: List[str] = ["tree", "bushes"]

class PanosDetectObjectsRequest(BaseModel):
    lat: float
    lon: float
    count: int = 3
    area_of_interest: float = 100.0
    min_distance: float = 10.0
    labels: List[str] = ["tree", "bushes"]

class ClassifyPlantRequest(BaseModel):
    image_path: str

class PanosDetectAndClassifyRequest(BaseModel):
    lat: float
    lon: float
    count: int = 3
    area_of_interest: float = 100.0
    min_distance: float = 10.0
    labels: List[str] = ["tree", "bushes"]

class LandcoverRequest(BaseModel):
    lon: float
    lat: float
    buffer_km: float = 3.0
    date_start: str = "2024-10-01"
    date_end: str = "2024-11-15"
    scale: int = 10
    cloud_cover_max: int = 20

class VegetationRequest(BaseModel):
    lon: float
    lat: float
    buffer_km: float = 3.0
    mask_path: Optional[str] = None

class LandcoverVegetationRequest(BaseModel):
    lon: float
    lat: float
    buffer_km: float = 3.0
    date_start: str = "2024-10-01"
    date_end: str = "2024-11-15"
    scale: int = 10
    cloud_cover_max: int = 20

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

# -----------------------------------------------------------------------------
# Security & Auth Helpers
# -----------------------------------------------------------------------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    user_id = payload.get("sub")
    userid = payload.get("userid")
    role = payload.get("role")
    
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    return {"user_id": user_id, "userid": userid, "role": role}

async def get_current_admin(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user

# -----------------------------------------------------------------------------
# Auth Endpoints
# -----------------------------------------------------------------------------
@app.post("/auth/register")
async def register(user: UserRegister):
    """Register a new user"""
    if user.role not in ["user", "admin"]:
        user.role = "user"
    
    password_hash = get_password_hash(user.password)
    user_id = generate_uuid()
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (id, userid, name, password_hash, role) VALUES (?, ?, ?, ?, ?)",
            (user_id, user.userid, user.name, password_hash, user.role)
        )
        conn.commit()
        conn.close()
        return {"message": "Registration successful", "userid": user.userid}
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(status_code=400, detail="Username already exists")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    """Login and get JWT token"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, password_hash, role FROM users WHERE userid = ?", (user.userid,))
    db_user = cur.fetchone()
    conn.close()
    
    if not db_user or not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    user_id = db_user["id"]
    role = db_user["role"]
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_id, "userid": user.userid, "role": role},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": role,
        "userid": user.userid
    }

@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info"""
    score = get_user_score(current_user["user_id"])
    return {
        "userid": current_user["userid"],
        "role": current_user["role"],
        "score": score
    }

# -----------------------------------------------------------------------------
# Upload Endpoints
# -----------------------------------------------------------------------------
@app.post("/upload")
async def upload_image(
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    species: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """Upload an image with geolocation"""
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image selected")
    
    image_bytes = await image.read()
    filename = image.filename
    content_type = image.content_type
    upload_id = generate_uuid()
    user_id = current_user["user_id"]
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO uploads (id, user_id, filename, content_type, image, latitude, longitude, species)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (upload_id, user_id, filename, content_type, image_bytes, latitude, longitude, species)
    )
    conn.commit()
    conn.close()
    
    return {
        "message": "Upload successful",
        "id": upload_id,
        "filename": filename,
        "latitude": latitude,
        "longitude": longitude,
        "species": species
    }

@app.get("/uploads/me")
async def get_my_uploads(current_user: dict = Depends(get_current_user), limit: Optional[int] = None):
    """Get all uploads by current user"""
    uploads = get_user_uploads(current_user["user_id"], limit)
    return {"uploads": uploads}

@app.post("/uploads/search")
async def search_uploads(req: UploadSearchRequest, current_user: dict = Depends(get_current_user)):
    """Search uploads within a radius (Community Search)"""
    # Pass None as user_id to search globally, not just for the current user
    uploads = get_uploads_in_radius(req.latitude, req.longitude, req.radius_km, None) 
    return {
        "uploads": uploads,
        "count": len(uploads),
        "search_params": {
            "latitude": req.latitude,
            "longitude": req.longitude,
            "radius_km": req.radius_km
        }
    }

@app.get("/image/{image_id}")
async def get_image(image_id: str, current_user: dict = Depends(get_current_user)):
    """Get an uploaded image by ID"""
    conn = get_connection()
    cur = conn.cursor()
    
    # Allow any authenticated user to view the image (removed 'AND user_id = ?')
    cur.execute("SELECT image, content_type FROM uploads WHERE id = ?", (image_id,))
    
    result = cur.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return StreamingResponse(iter([result["image"]]), media_type=result["content_type"])
# -----------------------------------------------------------------------------
# Pipeline Endpoints (unchanged from original, but now protected)
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "AgniVed Pipeline API - Use /docs for API documentation"}

@app.post('/panos')
async def panos_api(request: PanosRequest, current_user: dict = Depends(get_current_user)):
    result = core.find_panos_and_views(
        lat=request.lat,
        lon=request.lon,
        n=request.count,
        radius_m=request.area_of_interest,
        min_distance_m=request.min_distance
    )
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'panos_result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    return result

@app.post('/detect_objects')
async def detect_objects_api(request: DetectObjectsRequest, current_user: dict = Depends(get_current_user)):
    result = run_object_detection(request.image_path, request.labels)
    return result

@app.post('/panos_detect_objects')
async def panos_detect_objects_api(request: PanosDetectObjectsRequest, current_user: dict = Depends(get_current_user)):
    pano_result = core.find_panos_and_views(
        lat=request.lat,
        lon=request.lon,
        n=request.count,
        radius_m=request.area_of_interest,
        min_distance_m=request.min_distance
    )
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'panos_detect_objects_result.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(pano_result, f, indent=2)

    detected_objects = []
    for img in pano_result.get('images', []):
        if img.get('pano_downloaded'):
            pano_path = img.get('pano_path')
            if pano_path and os.path.exists(pano_path):
                detect_result = run_object_detection(pano_path, request.labels)
                detected_objects.append({
                    'pano_id': img.get('id'),
                    'pano_path': pano_path,
                    'object_detection': detect_result
                })
            else:
                detected_objects.append({
                    'pano_id': img.get('id'),
                    'pano_path': pano_path,
                    'object_detection': 'Pano image not found.'
                })
        else:
            detected_objects.append({
                'pano_id': img.get('id'),
                'pano_path': img.get('pano_path'),
                'object_detection': 'Pano not downloaded.'
            })

    combined_result = {
        'pano_result': pano_result,
        'detected_objects': detected_objects
    }
    return combined_result

@app.post('/classify_plant')
async def classify_plant_api(request: ClassifyPlantRequest, current_user: dict = Depends(get_current_user)):
    if not request.image_path or not os.path.exists(request.image_path):
        raise HTTPException(status_code=400, detail="image_path not provided or file does not exist")
    try:
        image = Image.open(request.image_path)
        inputs = plant_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = plant_model(**inputs).logits
        pred = logits.softmax(dim=-1)[0]
        topk = torch.topk(pred, k=5)
        results = []
        for prob, idx in zip(topk.values, topk.indices):
            label = plant_model.config.id2label[idx.item()]
            results.append({"label": label, "probability": float(prob.item())})
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/panos_detect_and_classify')
async def panos_detect_and_classify_api(request: PanosDetectAndClassifyRequest, current_user: dict = Depends(get_current_user)):
    pano_result = core.find_panos_and_views(
        lat=request.lat,
        lon=request.lon,
        n=request.count,
        radius_m=request.area_of_interest,
        min_distance_m=request.min_distance
    )
    out_dir = os.path.join(os.path.dirname(__file__), 'out')
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'panos_detect_and_classify_result.json')

    detected_objects = []
    classify_results = []

    for img in pano_result.get('images', []):
        if img.get('pano_downloaded'):
            pano_path = img.get('pano_path')
            if pano_path and os.path.exists(pano_path):
                detect_result = run_object_detection(pano_path, request.labels)
                detected_objects.append({
                    'pano_id': img.get('id'),
                    'pano_path': pano_path,
                    'object_detection': detect_result
                })

                crop_dir = os.path.join(os.path.dirname(pano_path), '..', 'detected_crops')
                crop_dir = os.path.abspath(crop_dir)
                crop_images = glob.glob(os.path.join(crop_dir, '*.jpg'))
                crop_classifications = []
                for crop_img in crop_images:
                    try:
                        image = Image.open(crop_img)
                        inputs = plant_processor(images=image, return_tensors="pt")
                        with torch.no_grad():
                            logits = plant_model(**inputs).logits
                        pred = logits.softmax(dim=-1)[0]
                        topk = torch.topk(pred, k=5)
                        results = []
                        for prob, idx in zip(topk.values, topk.indices):
                            label = plant_model.config.id2label[idx.item()]
                            results.append({"label": label, "probability": float(prob.item())})
                        crop_classifications.append({
                            "crop_image": crop_img,
                            "classification": results
                        })
                    except Exception as e:
                        crop_classifications.append({
                            "crop_image": crop_img,
                            "error": str(e)
                        })
                classify_results.append({
                    'pano_id': img.get('id'),
                    'crop_classifications': crop_classifications
                })
            else:
                detected_objects.append({
                    'pano_id': img.get('id'),
                    'pano_path': pano_path,
                    'object_detection': 'Pano image not found.'
                })
        else:
            detected_objects.append({
                'pano_id': img.get('id'),
                'pano_path': img.get('pano_path'),
                'object_detection': 'Pano not downloaded.'
            })

    combined_result = {
        'pano_result': pano_result,
        'detected_objects': detected_objects,
        'classify_results': classify_results
    }
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(combined_result, f, indent=2)
    return combined_result

@app.post('/run_landcover')
async def run_landcover(request: LandcoverRequest, current_user: dict = Depends(get_current_user)):
    try:
        parent_dir = Path(__file__).resolve().parent
        results_dir = parent_dir / "LandcoverResults"
        aoi_cfg = LandcoverAOIConfig(lon=request.lon, lat=request.lat, buffer_km=request.buffer_km)
        dl_cfg = DownloadConfig(
            output_dir=results_dir,
            date_start=request.date_start,
            date_end=request.date_end,
            scale=request.scale,
            cloud_cover_max=request.cloud_cover_max,
        )
        outputs = run_landcover_pipeline(aoi_cfg, dl_cfg)
        
        # --- FIX: Convert visual output to Base64 ---
        viz_path = outputs.get('viz_path')
        if not viz_path:
             potential_path = results_dir / "agnived_cover_analysis.png"
             if potential_path.exists():
                 viz_path = str(potential_path)

        b64_string = image_to_base64(str(viz_path))
        
        return {
            **{k: str(v) for k, v in outputs.items()},
            "image_base64": b64_string # Send Base64 data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={'error': str(e), 'trace': traceback.format_exc()})

@app.post('/run_vegetation')
async def run_vegetation(request: VegetationRequest, current_user: dict = Depends(get_current_user)):
    try:
        mask_path = request.mask_path
        if not mask_path:
            parent_dir = Path(__file__).resolve().parent
            mask_path = str(parent_dir / "LandcoverResults" / "vegetation_mask.tif")
        
        aoi_cfg = VegAOIConfig(lon=request.lon, lat=request.lat, buffer_km=request.buffer_km)
        res = run_bigearth_rdnet(aoi_cfg, veg_mask_path=Path(mask_path))
        
        # --- FIX: Convert visual output to Base64 ---
        b64_string = image_to_base64(str(res.viz_path))

        result = {
            'aoi': vars(res.aoi) if hasattr(res.aoi, '__dict__') else str(res.aoi),
            'cube_path': str(res.cube_path),
            'viz_path': str(res.viz_path),
            'class_distribution': res.class_distribution,
            'tile_counts': res.tile_counts,
            'avg_confidence': res.avg_confidence,
            'image_base64': b64_string # Send Base64 data
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={'error': str(e), 'trace': traceback.format_exc()})

# Code marked for Removal

# @app.post('/run_landcover_and_vegetation')
# async def run_landcover_and_vegetation(request: LandcoverVegetationRequest, current_user: dict = Depends(get_current_user)):
#     try:
#         parent_dir = Path(__file__).resolve().parent
#         results_dir = parent_dir / "LandcoverResults"
#         aoi_cfg = LandcoverAOIConfig(lon=request.lon, lat=request.lat, buffer_km=request.buffer_km)
#         dl_cfg = DownloadConfig(
#             output_dir=results_dir,
#             date_start=request.date_start,
#             date_end=request.date_end,
#             scale=request.scale,
#             cloud_cover_max=request.cloud_cover_max,
#         )
#         landcover_outputs = run_landcover_pipeline(aoi_cfg, dl_cfg)
        
#         mask_path = landcover_outputs.get('vegetation_mask')
#         veg_aoi_cfg = VegAOIConfig(lon=request.lon, lat=request.lat, buffer_km=request.buffer_km)
#         veg_res = run_bigearth_rdnet(veg_aoi_cfg, veg_mask_path=Path(mask_path))
        
#         # Base64 Conversions
#         lc_viz = landcover_outputs.get('viz_path') or str(results_dir / "agnived_cover_analysis.png")
#         veg_viz = str(veg_res.viz_path)
        
#         landcover_b64 = image_to_base64(lc_viz)
#         veg_b64 = image_to_base64(veg_viz)

#         veg_result = {
#             'aoi': vars(veg_res.aoi) if hasattr(veg_res.aoi, '__dict__') else str(veg_res.aoi),
#             'cube_path': str(veg_res.cube_path),
#             'viz_path': str(veg_res.viz_path),
#             'class_distribution': veg_res.class_distribution,
#             'tile_counts': veg_res.tile_counts,
#             'avg_confidence': veg_res.avg_confidence,
#             'image_base64': veg_b64
#         }
        
#         return {
#             'landcover': {
#                 **{k: str(v) for k, v in landcover_outputs.items()},
#                 'image_base64': landcover_b64
#             },
#             'vegetation': veg_result
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail={'error': str(e), 'trace': traceback.format_exc()})

# @app.post('/run_landcover_vegetation_and_panos')
# async def run_landcover_vegetation_and_panos(request: LandcoverVegetationPanosRequest, current_user: dict = Depends(get_current_user)):
#     try:
#         parent_dir = Path(__file__).resolve().parent
#         results_dir = parent_dir / "LandcoverResults"
#         aoi_cfg = LandcoverAOIConfig(lon=request.lon, lat=request.lat, buffer_km=request.buffer_km)
#         dl_cfg = DownloadConfig(
#             output_dir=results_dir,
#             date_start=request.date_start,
#             date_end=request.date_end,
#             scale=request.scale,
#             cloud_cover_max=request.cloud_cover_max,
#         )
#         landcover_outputs = run_landcover_pipeline(aoi_cfg, dl_cfg)
#         mask_path = landcover_outputs.get('vegetation_mask')
#         veg_aoi_cfg = VegAOIConfig(lon=request.lon, lat=request.lat, buffer_km=request.buffer_km)
#         veg_res = run_bigearth_rdnet(veg_aoi_cfg, veg_mask_path=Path(mask_path))
        
#         veg_result = {
#             'aoi': vars(veg_res.aoi) if hasattr(veg_res.aoi, '__dict__') else str(veg_res.aoi),
#             'cube_path': str(veg_res.cube_path),
#             'viz_path': str(veg_res.viz_path),
#             'class_distribution': veg_res.class_distribution,
#             'tile_counts': veg_res.tile_counts,
#             'avg_confidence': veg_res.avg_confidence,
#         }
#         panos_lat = request.panos_lat if request.panos_lat is not None else request.lat
#         panos_lon = request.panos_lon if request.panos_lon is not None else request.lon
#         pano_result = core.find_panos_and_views(
#             lat=panos_lat,
#             lon=panos_lon,
#             n=request.panos_count,
#             radius_m=request.panos_area_of_interest,
#             min_distance_m=request.panos_min_distance
#         )
#         detected_objects = []
#         classify_results = []
#         for img in pano_result.get('images', []):
#             if img.get('pano_downloaded'):
#                 pano_path = img.get('pano_path')
#                 if pano_path and os.path.exists(pano_path):
#                     detect_result = run_object_detection(pano_path, request.panos_labels)
#                     detected_objects.append({
#                         'pano_id': img.get('id'),
#                         'pano_path': pano_path,
#                         'object_detection': detect_result
#                     })
#                     crop_dir = os.path.join(os.path.dirname(pano_path), '..', 'detected_crops')
#                     crop_dir = os.path.abspath(crop_dir)
#                     crop_images = glob.glob(os.path.join(crop_dir, '*.jpg'))
#                     crop_classifications = []
#                     for crop_img in crop_images:
#                         try:
#                             image = Image.open(crop_img)
#                             inputs = plant_processor(images=image, return_tensors="pt")
#                             with torch.no_grad():
#                                 logits = plant_model(**inputs).logits
#                             pred = logits.softmax(dim=-1)[0]
#                             topk = torch.topk(pred, k=5)
#                             results = []
#                             for prob, idx in zip(topk.values, topk.indices):
#                                 label = plant_model.config.id2label[idx.item()]
#                                 results.append({"label": label, "probability": float(prob.item())})
#                             crop_classifications.append({
#                                 "crop_image": crop_img,
#                                 "classification": results
#                             })
#                         except Exception as e:
#                             crop_classifications.append({
#                                 "crop_image": crop_img,
#                                 "error": str(e)
#                             })
#                     classify_results.append({
#                         'pano_id': img.get('id'),
#                         'crop_classifications': crop_classifications
#                     })
#                 else:
#                     detected_objects.append({
#                         'pano_id': img.get('id'),
#                         'pano_path': pano_path,
#                         'object_detection': 'Pano image not found.'
#                     })
#             else:
#                 detected_objects.append({
#                     'pano_id': img.get('id'),
#                     'pano_path': img.get('pano_path'),
#                     'object_detection': 'Pano not downloaded.'
#                 })
#         panos_combined_result = {
#             'pano_result': pano_result,
#             'detected_objects': detected_objects,
#             'classify_results': classify_results
#         }
#         result = {
#             'landcover': {k: str(v) for k, v in landcover_outputs.items()},
#             'vegetation': veg_result,
#             'panos': panos_combined_result
#         }
#         out_dir = os.path.join(os.path.dirname(__file__), 'out')
#         os.makedirs(out_dir, exist_ok=True)
#         result_path = os.path.join(out_dir, 'landcover_vegetation_and_panos_result.json')
#         with open(result_path, 'w', encoding='utf-8') as f:
#             json.dump(result, f, indent=2)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail={'error': str(e), 'trace': traceback.format_exc()})

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------


@app.post('/run_panos_and_plant_identification')
async def run_panos_and_plant_identification(request: LandcoverVegetationPanosRequest):
    """
    Fast pipeline: Panos + Object Detection + Plant Identification for all detected crops
    Skips landcover and vegetation analysis - focuses only on street-level imagery analysis
    Returns base64-encoded images for frontend storage
    """
    try:
        # Panos detect & identify plants
        panos_lat = request.panos_lat if request.panos_lat is not None else request.lat
        panos_lon = request.panos_lon if request.panos_lon is not None else request.lon
        
        pano_result = find_panos_and_views(
            lat=panos_lat,
            lon=panos_lon,
            n=request.panos_count,
            radius_m=request.panos_area_of_interest,
            min_distance_m=request.panos_min_distance
        )
        
        detected_objects = []
        plant_identification_results = []
        
        for img in pano_result.get('images', []):
            if img.get('pano_downloaded'):
                pano_path = img.get('pano_path')
                if pano_path and os.path.exists(pano_path):
                    detect_result = run_object_detection(pano_path, request.panos_labels)
                    
                    # Encode panorama image to base64
                    pano_base64 = image_to_base64(pano_path)
                    
                    detected_objects.append({
                        'pano_id': img.get('id'),
                        'pano_path': pano_path,
                        'pano_image_base64': pano_base64,
                        'object_detection': detect_result
                    })
                    
                    # Get crop images from the detection result
                    crops = detect_result.get('crops', [])
                    crop_plant_identifications = []
                    
                    # Run identify_plant for each detected crop
                    for crop_info in crops:
                        crop_path = crop_info.get('crop_path')
                        if not crop_path:
                            continue
                            
                        # Make absolute path if relative
                        if not os.path.isabs(crop_path):
                            crop_path = os.path.abspath(crop_path)
                        
                        if not os.path.exists(crop_path):
                            crop_plant_identifications.append({
                                "crop_image": crop_path,
                                "label": crop_info.get('label'),
                                "score": crop_info.get('score'),
                                "error": f"Crop file not found: {crop_path}"
                            })
                            continue
                        
                        try:
                            # Call identify_plant logic for this crop
                            image = Image.open(crop_path)
                            inputs = plant_processor(images=image, return_tensors="pt")
                            # Move inputs to the same device as the model
                            inputs = {key: val.to(plant_model.device) for key, val in inputs.items()}
                            with torch.no_grad():
                                logits = plant_model(**inputs).logits
                            pred = logits.softmax(dim=-1)[0]
                            topk = torch.topk(pred, k=5)
                            predictions = []
                            for prob, idx in zip(topk.values, topk.indices):
                                label = plant_model.config.id2label[idx.item()]
                                predictions.append({
                                    "species": label,
                                    "confidence": float(prob.item()),
                                    "confidence_percentage": f"{prob.item() * 100:.2f}%"
                                })
                            
                            # Encode crop image to base64
                            crop_base64 = image_to_base64(crop_path)
                            
                            # Plant identification result with species info and base64 image
                            plant_id_result = {
                                "crop_image": crop_path,
                                "crop_image_base64": crop_base64,
                                "object_label": crop_info.get('label'),
                                "object_score": crop_info.get('score'),
                                "predictions": predictions,
                                "top_prediction": predictions[0] if predictions else None
                            }
                            crop_plant_identifications.append(plant_id_result)
                            
                        except Exception as e:
                            crop_plant_identifications.append({
                                "crop_image": crop_path,
                                "object_label": crop_info.get('label'),
                                "object_score": crop_info.get('score'),
                                "error": str(e)
                            })
                    
                    plant_identification_results.append({
                        'pano_id': img.get('id'),
                        'pano_path': pano_path,
                        'pano_image_base64': pano_base64,
                        'crop_plant_identifications': crop_plant_identifications,
                        'total_crops_identified': len([c for c in crop_plant_identifications if 'predictions' in c])
                    })
                else:
                    detected_objects.append({
                        'pano_id': img.get('id'),
                        'pano_path': pano_path,
                        'object_detection': 'Pano image not found.'
                    })
                    plant_identification_results.append({
                        'pano_id': img.get('id'),
                        'pano_path': pano_path,
                        'error': 'Pano image not found.'
                    })
            else:
                detected_objects.append({
                    'pano_id': img.get('id'),
                    'pano_path': img.get('pano_path'),
                    'object_detection': 'Pano not downloaded.'
                })
                plant_identification_results.append({
                    'pano_id': img.get('id'),
                    'pano_path': img.get('pano_path'),
                    'error': 'Pano not downloaded.'
                })
        
        panos_combined_result = {
            'pano_result': pano_result,
            'detected_objects': detected_objects,
            'plant_identification_results': plant_identification_results
        }
        
        result = {
            'panos': panos_combined_result,
            'summary': {
                'total_panos': pano_result.get('count_returned', 0),
                'total_objects_detected': sum(len(obj.get('object_detection', {}).get('crops', [])) for obj in detected_objects if isinstance(obj.get('object_detection'), dict)),
                'total_plants_identified': sum(len([c for c in pir.get('crop_plant_identifications', []) if 'predictions' in c]) for pir in plant_identification_results)
            }
        }
        
        # Save result to Pano360OutputDir/panos_and_plant_identification_result.json
        out_dir = os.path.join(os.path.dirname(__file__), 'Pano360OutputDir')
        os.makedirs(out_dir, exist_ok=True)
        result_path = os.path.join(out_dir, 'panos_and_plant_identification_result.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={'error': str(e), 'trace': traceback.format_exc()})



if __name__ == '__main__':
    import uvicorn
    uvicorn.run("server_fastapi:app", host='0.0.0.0', port=5000, reload=True)