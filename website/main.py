import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

IMAGE_ROOT = Path(os.environ.get("IMAGE_ROOT", "/data"))

app = FastAPI()


@app.get("/api/cameras")
def list_cameras():
    if not IMAGE_ROOT.exists():
        return []
    return sorted(p.name for p in IMAGE_ROOT.iterdir() if p.is_dir())
