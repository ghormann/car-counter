import os
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

IMAGE_ROOT = Path(os.environ.get("IMAGE_ROOT", "/data"))
PAGE_SIZE = 50

app = FastAPI()


def _require_dir(path: Path) -> Path:
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"{path} not found")
    return path


@app.get("/api/cameras")
def list_cameras():
    if not IMAGE_ROOT.exists():
        return []
    return sorted(p.name for p in IMAGE_ROOT.iterdir() if p.is_dir())


@app.get("/api/years")
def list_years(camera: str):
    base = _require_dir(IMAGE_ROOT / camera)
    return sorted(p.name for p in base.iterdir() if p.is_dir())


@app.get("/api/months")
def list_months(camera: str, year: str):
    base = _require_dir(IMAGE_ROOT / camera / year)
    return sorted(p.name for p in base.iterdir() if p.is_dir())


@app.get("/api/days")
def list_days(camera: str, year: str, month: str):
    base = _require_dir(IMAGE_ROOT / camera / year / month)
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def _parse_ts(filename: str) -> str:
    """Extract sortable timestamp string from filename."""
    # startup_YYYYMMDD_HHMMSS.jpg  or  YYYYMMDD_HHMMSS.jpg
    m = re.search(r"(\d{8}_\d{6})", filename)
    return m.group(1) if m else filename


@app.get("/api/images")
def list_images(
    camera: str,
    year: str,
    month: str,
    day: str,
    sort: str = Query(default="desc", pattern="^(asc|desc)$"),
    page: int = Query(default=1, ge=1),
):
    base = _require_dir(IMAGE_ROOT / camera / year / month / day)
    files = sorted(
        [p.name for p in base.iterdir() if p.suffix.lower() == ".jpg"],
        key=_parse_ts,
        reverse=(sort == "desc"),
    )
    total = len(files)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    start = (page - 1) * PAGE_SIZE
    page_files = files[start : start + PAGE_SIZE]

    def _human_ts(filename: str) -> str:
        m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", filename)
        if not m:
            return filename
        Y, Mo, D, h, mi, s = m.groups()
        return f"{Y}-{Mo}-{D} {h}:{mi}:{s}"

    images = [
        {
            "filename": f,
            "timestamp": _human_ts(f),
            "is_startup": f.startswith("startup_"),
            "url": f"/images/{camera}/{year}/{month}/{day}/{f}",
        }
        for f in page_files
    ]
    return {"page": page, "total_pages": total_pages, "total": total, "images": images}
