import sys
import os
import io
import json
import pickle
import numpy as np
import torch
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from facenet_pytorch import MTCNN

# -----------------------------------------------------------------------------
# PATH FIX: Add the parent directory ('backend') to sys.path
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../backend/api
backend_dir = os.path.dirname(current_dir)             # .../backend
project_root = os.path.dirname(backend_dir)            # .../adversarial-face-patch (Root)
sys.path.append(backend_dir)
sys.path.append(project_root) # Add root to find src if needed
# -----------------------------------------------------------------------------

from src.models.face_recognition import FaceRecognitionModel
from src.utils.config import load_config, get_device
from src.attack.patch_application import load_patch, apply_circular_patch, create_circular_mask
from src.data.dataset import load_saved_images

app = FastAPI(title="Face Recognition Security Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
class State:
    config = None
    device = None
    face_model = None
    mtcnn = None
    metadata = None
    patches = {}
    attacker_images = {} 

state = State()

def resolve_path(relative_path):
    """
    Tries to find a file path by checking:
    1. Exact path defined in config
    2. Inside 'backend' folder
    3. Inside project root folder
    """
    if os.path.exists(relative_path):
        return relative_path
    
    # Check inside backend/
    path_in_backend = os.path.join(backend_dir, relative_path)
    if os.path.exists(path_in_backend):
        return path_in_backend
        
    # Check in project root (../data instead of backend/data)
    path_in_root = os.path.join(project_root, relative_path)
    if os.path.exists(path_in_root):
        return path_in_root
        
    return relative_path # Return original if not found (will fail later)

@app.on_event("startup")
async def startup_event():
    print("Loading System Resources...")
    
    # 1. Load Config
    try:
        # Try standard load
        state.config = load_config()
    except Exception:
        # Fallback manual load
        config_path = resolve_path("backend/config/config.yaml")
        if not os.path.exists(config_path):
             config_path = resolve_path("config/config.yaml")
             
        if os.path.exists(config_path):
            print(f"Found config at: {config_path}")
            import yaml
            with open(config_path, 'r') as f:
                state.config = yaml.safe_load(f)
        else:
            print("CRITICAL WARNING: Could not find config.yaml")
            return

    state.device = get_device()

    # 2. Load Model & Database
    state.face_model = FaceRecognitionModel(device=state.device)
    
    # Resolve DB path using the new robust function
    raw_db_path = state.config['employee_db_path'] # e.g., "data/processed/..."
    db_path = resolve_path(raw_db_path)

    if os.path.exists(db_path):
        print(f"Loading database from: {db_path}")
        state.face_model.load_employee_database(db_path)
        
        # Load Metadata
        metadata_path = db_path.replace('.pkl', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                state.metadata = pickle.load(f)
                print(f"Loaded metadata for {len(state.metadata['target_names'])} people.")
    else:
        print(f"WARNING: Employee DB not found at {db_path}.")
        print(f"   PLEASE RUN 'python 02_build_employee_db.py' FROM THE PROJECT ROOT FIRST.")
    
    # 3. Initialize MTCNN
    state.mtcnn = MTCNN(image_size=160, margin=20, device=state.device, keep_all=False)

    # 4. Pre-load Attacker Images
    if state.metadata:
        raw_dir_config = state.config["raw_data_dir"]
        raw_dir = resolve_path(raw_dir_config)
            
        if os.path.exists(raw_dir):
            attacker_ids = state.metadata['attacker_ids']
            attacker_names = [state.metadata['target_names'][i] for i in attacker_ids]
            
            print(f"Pre-loading {len(attacker_names)} attackers...")
            # We need to be careful with load_saved_images path construction
            # It usually appends "attackers" to the path passed
            attackers_path = os.path.join(raw_dir, "attackers")
            if os.path.exists(attackers_path):
                loaded_imgs = load_saved_images(attackers_path, attacker_names)
                for name, imgs in zip(attacker_names, loaded_imgs):
                    if len(imgs) > 0:
                        state.attacker_images[name] = imgs[0]
            else:
                 print(f"WARNING: Attackers folder not found at {attackers_path}")
        else:
            print(f"WARNING: Raw data directory not found at {raw_dir}")

    # 5. Pre-load Patches
    patches_dir_config = state.config['patches_dir']
    patches_dir = resolve_path(patches_dir_config)

    if os.path.exists(patches_dir):
        for f in os.listdir(patches_dir):
            if f.endswith('.pt'):
                state.patches[f] = os.path.join(patches_dir, f)
    
    print("System Ready.")

@app.get("/info")
async def get_system_info():
    """Returns available attackers and employees."""
    if not state.metadata:
        raise HTTPException(
            status_code=503, 
            detail="System not initialized. Database file missing. Run 02_build_employee_db.py."
        )
    
    target_names = state.metadata['target_names']
    
    attackers = [
        {"id": idx, "name": target_names[idx], "db_id": f"att_{idx}"} 
        for idx in state.metadata['attacker_ids']
    ]
    
    employees = [
        {"id": idx, "name": target_names[idx], "db_id": f"emp_{idx}"} 
        for idx in state.metadata['employee_ids']
    ]
    
    return {
        "attackers": attackers,
        "employees": employees
    }

@app.get("/image")
async def get_image(attacker_id: int, target_id: int = None, mode: str = Query("raw", enum=["raw", "patched"])):
    """
    Returns a cropped 160x160 face image.
    Both 'raw' and 'patched' go through detection to ensure consistent Aspect Ratio.
    """
    if not state.metadata:
        raise HTTPException(status_code=503, detail="System not ready")

    try:
        attacker_name = state.metadata['target_names'][attacker_id]
    except IndexError:
        raise HTTPException(status_code=404, detail="Attacker ID not found")
    
    if attacker_name not in state.attacker_images:
        # Placeholder black image
        img = Image.new('RGB', (160, 160), color='black')
    else:
        original_img = state.attacker_images[attacker_name]
        
        # 1. Detect Face (This converts it to a 160x160 tensor)
        face_tensor = state.mtcnn(original_img)
        
        # If detection fails, fallback to original (which might be rectangular)
        if face_tensor is None:
             final_img_pil = original_img.resize((160, 160)) # Force resize fallback
        else:
            face_tensor = face_tensor.to(state.device)
            
            # 2. Apply Patch if mode is patched
            if mode == "patched" and target_id is not None and state.patches:
                patch_path = list(state.patches.values())[0] 
                patch_tensor, patch_meta = load_patch(patch_path)
                
                if patch_tensor is not None:
                    patch_tensor = patch_tensor.to(state.device)
                    size = patch_meta.get('size', 70)
                    mask = create_circular_mask(size, size // 2, state.device) if patch_meta.get('type') == 'circular' else torch.ones(1, size, size).to(state.device)
                    
                    # Overwrite face_tensor with patched version
                    face_tensor = apply_circular_patch(face_tensor, patch_tensor, patch_meta['position']['x'], patch_meta['position']['y'], mask)
            
            # 3. Convert Tensor back to Image for display
            # Tensor is [3, 160, 160]
            display_np = face_tensor.permute(1, 2, 0).cpu().detach().numpy()
            
            # Normalize to 0-255 visually
            if display_np.max() > display_np.min():
                display_np = (display_np - display_np.min()) / (display_np.max() - display_np.min())
            
            display_np = (display_np * 255).astype(np.uint8)
            final_img_pil = Image.fromarray(display_np)
        
        img = final_img_pil

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type="image/png")

@app.post("/scan")
async def scan_face(payload: dict):
    attacker_id = payload.get("attacker_id")
    mode = payload.get("mode")
    defense_enabled = payload.get("defense")
    
    if not state.metadata or attacker_id is None:
        return {"status": "error", "message": "SYSTEM ERROR", "detail": "Metadata Missing", "confidence": 0, "color": "red"}

    try:
        attacker_name = state.metadata['target_names'][attacker_id]
    except IndexError:
        return {"status": "error", "message": "SYSTEM ERROR", "detail": "Invalid ID", "confidence": 0, "color": "red"}
    
    if attacker_name not in state.attacker_images:
         return {"status": "error", "message": "IMAGE ERROR", "detail": "Image Not Found", "confidence": 0, "color": "red"}

    original_img = state.attacker_images[attacker_name]
    face_tensor = state.mtcnn(original_img)
    
    if face_tensor is None:
        return {"status": "error", "message": "NO FACE", "detail": "Adjust Camera", "confidence": 0, "color": "red"}
        
    face_tensor = face_tensor.to(state.device)
    
    if mode == "patched" and state.patches:
        patch_path = list(state.patches.values())[0]
        patch_tensor, patch_meta = load_patch(patch_path)
        size = patch_meta.get('size', 70)
        mask = create_circular_mask(size, size // 2, state.device) if patch_meta.get('type') == 'circular' else torch.ones(1, size, size).to(state.device)
        patch_tensor = patch_tensor.to(state.device)
        face_tensor = apply_circular_patch(face_tensor, patch_tensor, patch_meta['position']['x'], patch_meta['position']['y'], mask)
        
        if defense_enabled:
             return {"status": "threat", "message": "THREAT DETECTED", "detail": "ADVERSARIAL PATTERN", "subtext": "Blocked by Defense Layer", "confidence": 99, "color": "red"}

    if len(face_tensor.shape) == 3:
        face_tensor = face_tensor.unsqueeze(0)
        
    embedding = state.face_model.model(face_tensor)
    identified_name, confidence = state.face_model.classify_face(embedding, threshold=state.config['classification_threshold'])
    
    if identified_name == "Unknown":
        return {"status": "denied", "message": "ACCESS DENIED", "detail": "UNRECOGNIZED IDENTITY", "confidence": round(confidence * 100, 1), "color": "red"}
    else:
        return {"status": "granted", "message": "ACCESS GRANTED", "detail": identified_name.upper(), "subtext": "ENGINEERING DEPT", "confidence": round(confidence * 100, 1), "color": "green"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)