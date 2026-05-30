from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
import tempfile
import uuid
import asyncio

from pipeline import run_full_pipeline

app = FastAPI(title="VisionSafe AI Navigation API", version="1.0.0")

# ── CORS (allow your Vercel frontend + localhost dev) ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        os.getenv("FRONTEND_URL", ""),   # set this in Render env vars
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "/tmp/visionsafe_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "VisionSafe API is running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Main endpoint: receives an image, runs the full AI pipeline,
    returns JSON with safety label, direction, distance, explanation,
    narration text, plus paths to annotated image and audio.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png)")

    # Save upload to a temp file
    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    job_id = str(uuid.uuid4())[:8]
    tmp_input = os.path.join(OUTPUT_DIR, f"input_{job_id}{suffix}")
    tmp_img_out = os.path.join(OUTPUT_DIR, f"annotated_{job_id}.jpg")
    tmp_audio_out = os.path.join(OUTPUT_DIR, f"narration_{job_id}.mp3")

    try:
        contents = await file.read()
        with open(tmp_input, "wb") as f:
            f.write(contents)

        # Run the full pipeline (blocking — offloaded to thread pool)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            run_full_pipeline,
            tmp_input,
            tmp_img_out,
            tmp_audio_out,
        )

        return JSONResponse({
            "job_id": job_id,
            "label": result["label"],
            "direction": result["direction"],
            "distance_m": result["distance_m"],
            "confidence": result["confidence"],
            "explanation": result["explanation"],
            "narration_text": result["narration_text"],
            "annotated_image_url": f"/output/image/{job_id}",
            "audio_url": f"/output/audio/{job_id}",
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_input):
            os.remove(tmp_input)


@app.get("/output/image/{job_id}")
def get_annotated_image(job_id: str):
    path = os.path.join(OUTPUT_DIR, f"annotated_{job_id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/output/audio/{job_id}")
def get_audio(job_id: str):
    path = os.path.join(OUTPUT_DIR, f"narration_{job_id}.mp3")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/mpeg")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
