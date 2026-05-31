from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
import uuid
import asyncio
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visionsafe")

app = FastAPI(title="VisionSafe AI Navigation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png)")

    suffix = os.path.splitext(file.filename or "upload.jpg")[-1] or ".jpg"
    job_id = str(uuid.uuid4())[:8]
    tmp_input    = os.path.join(OUTPUT_DIR, f"input_{job_id}{suffix}")
    tmp_img_out  = os.path.join(OUTPUT_DIR, f"annotated_{job_id}.jpg")
    tmp_audio_out = os.path.join(OUTPUT_DIR, f"narration_{job_id}.mp3")

    try:
        contents = await file.read()
        logger.info(f"[{job_id}] Received image: {len(contents)//1024} KB")
        with open(tmp_input, "wb") as f:
            f.write(contents)

        # Import here (not at module level) so startup never loads torch
        from pipeline import run_full_pipeline

        logger.info(f"[{job_id}] Starting pipeline...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            run_full_pipeline,
            tmp_input,
            tmp_img_out,
            tmp_audio_out,
        )
        logger.info(f"[{job_id}] Pipeline complete: {result['label']} / {result['direction']}")

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

    except MemoryError:
        logger.error(f"[{job_id}] OUT OF MEMORY")
        raise HTTPException(status_code=503, detail="Out of memory — image too large or model too heavy for free tier")

    except Exception as e:
        logger.error(f"[{job_id}] Pipeline error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

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
