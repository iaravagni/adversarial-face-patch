"""
FastAPI application for adversarial face recognition system.
Provides endpoints for testing adversarial patches against face recognition models.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
import numpy as np
from PIL import Image
import io
import base64
import uvicorn


from src.models.face_recognition import FaceRecognitionModel
from src.attack.patch_application import apply_patch_to_image
from src.defense.patch_detection import PatchDetector
from src.utils.config import load_config

# Initialize FastAPI app
app = FastAPI(
    title="Adversarial Face Recognition API",
    description="API for testing adversarial patches against face recognition systems",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
config = load_config()
face_model = None
patch_detector = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Request/Response Models
class ScanRequest(BaseModel):
    image_data: str  # Base64 encoded image
    image_type: str  # 'raw' or 'patched'
    defense_enabled: bool
    patch_name: Optional[str] = None


class ScanResponse(BaseModel):
    status: str  # 'recognized', 'unknown', 'threat'
    message: str
    detail: str
    subtext: Optional[str] = None
    confidence: float
    color: str


class PatchInfo(BaseModel):
    name: str
    type: str  # 'circular', 'square', etc.
    size: int
    target_employee: str
    success_rate: float


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global face_model, patch_detector
    
    print(f"Using device: {device}")
    
    # Load face recognition model
    face_model = FaceRecognitionModel(device=device)
    face_model.load_employee_database(config['employee_db_path'])
    print("✓ Face recognition model loaded")
    
    # Load patch detector
    patch_detector = PatchDetector(device=device)
    print("✓ Patch detector loaded")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Adversarial Face Recognition API",
        "device": str(device)
    }


@app.get("/api/patches")
async def list_patches() -> List[PatchInfo]:
    """List all available adversarial patches"""
    import os
    import json
    
    patches_dir = config['patches_dir']
    patches = []
    
    for patch_file in os.listdir(patches_dir):
        if patch_file.endswith('.pt'):
            metadata_file = patch_file.replace('.pt', '_metadata.json')
            metadata_path = os.path.join(patches_dir, metadata_file)
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    patches.append(PatchInfo(**metadata))
    
    return patches


@app.post("/api/scan")
async def scan_face(request: ScanRequest) -> ScanResponse:
    """
    Scan a face image and return recognition results.
    
    Args:
        request: ScanRequest containing image data and configuration
        
    Returns:
        ScanResponse with recognition results
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        
        # Apply patch if needed
        if request.image_type == 'patched' and request.patch_name:
            patch_path = f"{config['patches_dir']}/{request.patch_name}.pt"
            image = apply_patch_to_image(image, patch_path)
        
        # Detect face and get embedding
        embedding = face_model.get_face_embedding(image)
        
        if embedding is None:
            return ScanResponse(
                status="error",
                message="NO FACE DETECTED",
                detail="Unable to detect face in image",
                confidence=0.0,
                color="red"
            )
        
        # Check for adversarial patterns if defense is enabled
        if request.defense_enabled:
            is_adversarial = patch_detector.detect_patch(image)
            
            if is_adversarial:
                return ScanResponse(
                    status="threat",
                    message="THREAT DETECTED",
                    detail="ADVERSARIAL PATTERN IDENTIFIED",
                    subtext="Patch-based attack blocked",
                    confidence=0.0,
                    color="red"
                )
        
        # Classify face
        identified, confidence = face_model.classify_face(
            embedding,
            threshold=config['classification_threshold']
        )
        
        if identified == "Unknown":
            return ScanResponse(
                status="unknown",
                message="ACCESS DENIED",
                detail="UNRECOGNIZED SUBJECT",
                confidence=0.0,
                color="red"
            )
        else:
            return ScanResponse(
                status="recognized",
                message="ACCESS GRANTED",
                detail=f"EMPLOYEE #{identified.split('_')[1]}",
                subtext=f"{identified.upper()} - ENGINEERING DEPT.",
                confidence=confidence * 100,
                color="green"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload and process an image"""
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to base64 for frontend
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "success": True,
            "image_data": f"data:image/png;base64,{img_str}",
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)