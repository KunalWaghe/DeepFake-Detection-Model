"""
DeepFake Detection API — FastAPI Backend
Provides a REST endpoint for analyzing uploaded videos.
"""

import os
import uuid
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model import DeepfakeDetector

# ─── Configuration ───────────────────────────────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Model Singleton ────────────────────────────────────────────────────────
detector = DeepfakeDetector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once at startup."""
    detector.load()
    yield
    # Cleanup on shutdown (if needed)


# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepFake Detection API",
    description="Upload a video to detect whether it is a deepfake or authentic.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend from any origin (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helper ──────────────────────────────────────────────────────────────────
def _get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    return {"status": "healthy", "model_loaded": detector.model is not None}


@app.post("/api/predict")
async def predict(video: UploadFile = File(...)):
    """
    Upload a video file and get a deepfake prediction.

    - **video**: Video file (mp4, avi, mov, mkv, webm). Max 100 MB.

    Returns JSON with `result` (FAKE/REAL) and `confidence` (0-100%).
    """
    # Validate file extension
    ext = _get_extension(video.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save uploaded file temporarily
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}.{ext}")

    try:
        # Stream the upload to disk (memory-efficient for large files)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Check file size after saving
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size // (1024*1024)}MB). Maximum is {MAX_FILE_SIZE // (1024*1024)}MB.",
            )

        # Run prediction
        result = detector.predict(temp_path)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Always clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ─── Run directly ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
