# Website Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight FastAPI + plain HTML/JS/CSS debug tool that browses annotated images saved to the car-counter PVC, served on port 8080.

**Architecture:** A single FastAPI process in `website/` serves both the REST API (listing cameras/years/months/days/images) and static files. The frontend is vanilla HTML/JS with no build step — it reads the API on load, populates cascading dropdowns, and renders a paginated thumbnail grid. Images are streamed directly from the PVC mount.

**Tech Stack:** Python 3.12+, FastAPI, uvicorn, plain HTML/JS/CSS, Docker (multi-arch), Kubernetes YAML, GitHub Actions.

---

## File Map

| File | Role |
|---|---|
| `website/main.py` | FastAPI app: all API endpoints + static file mount |
| `website/static/index.html` | Single-page shell: filter bar, grid, pagination |
| `website/static/app.js` | All client-side logic: dropdowns, fetch, render |
| `website/static/style.css` | Grid layout + thumbnail styling |
| `website/requirements.txt` | `fastapi`, `uvicorn[standard]` |
| `website/Dockerfile` | Multi-stage build, port 8080, `IMAGE_ROOT` env |
| `website/docker-compose.yml` | Local dev: mounts `../output` as `/data` |
| `tests/website/test_api.py` | Pytest tests for all API endpoints |
| `.github/workflows/docker-build-push.yml` | Add website build job |
| `docs/kubernetes-deployment.md` | Add website deployment section |

---

### Task 1: Python environment and FastAPI skeleton

**Files:**
- Create: `website/requirements.txt`
- Create: `website/main.py`
- Create: `tests/website/__init__.py`
- Create: `tests/website/test_api.py`

- [ ] **Step 1: Create `website/requirements.txt`**

```
fastapi==0.115.12
uvicorn[standard]==0.34.3
httpx==0.28.1
pytest==8.3.5
pytest-asyncio==0.26.0
```

- [ ] **Step 2: Write the failing test for `GET /api/cameras`**

Create `tests/website/__init__.py` (empty).

Create `tests/website/test_api.py`:

```python
import os
import pytest
from fastapi.testclient import TestClient

# Point IMAGE_ROOT at a temp fixture directory
@pytest.fixture(autouse=True)
def image_root(tmp_path, monkeypatch):
    # Layout: {root}/cam1/2024/01/15/  and  {root}/cam2/2024/02/01/
    (tmp_path / "cam1" / "2024" / "01" / "15").mkdir(parents=True)
    (tmp_path / "cam2" / "2024" / "02" / "01").mkdir(parents=True)
    # Add some image files for cam1/2024/01/15
    (tmp_path / "cam1" / "2024" / "01" / "15" / "20240115_120000.jpg").write_bytes(b"fake")
    (tmp_path / "cam1" / "2024" / "01" / "15" / "startup_20240115_060000.jpg").write_bytes(b"fake")
    (tmp_path / "cam1" / "2024" / "01" / "15" / "20240115_110000.jpg").write_bytes(b"fake")
    monkeypatch.setenv("IMAGE_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture()
def client(image_root):
    # Import after env var is set
    import importlib, website.main as m
    importlib.reload(m)
    return TestClient(m.app)


def test_cameras(client):
    r = client.get("/api/cameras")
    assert r.status_code == 200
    assert r.json() == ["cam1", "cam2"]
```

- [ ] **Step 3: Run test to confirm it fails**

```bash
cd /home/ghormann/src/car-counter
pip install -r website/requirements.txt
pytest tests/website/test_api.py::test_cameras -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `website.main` does not exist yet.

- [ ] **Step 4: Create `website/main.py` skeleton**

```python
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
```

Also create `website/__init__.py` (empty) so it is importable as a package:

```bash
touch website/__init__.py
```

- [ ] **Step 5: Run test to confirm it passes**

```bash
pytest tests/website/test_api.py::test_cameras -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add website/__init__.py website/main.py website/requirements.txt tests/website/
git commit -m "feat(website): FastAPI skeleton with /api/cameras endpoint"
```

---

### Task 2: Remaining listing API endpoints

**Files:**
- Modify: `website/main.py`
- Modify: `tests/website/test_api.py`

- [ ] **Step 1: Add tests for years, months, days, and images endpoints**

Append to `tests/website/test_api.py`:

```python
def test_years(client):
    r = client.get("/api/years?camera=cam1")
    assert r.status_code == 200
    assert r.json() == ["2024"]


def test_years_missing_camera(client):
    r = client.get("/api/years?camera=nope")
    assert r.status_code == 404


def test_months(client):
    r = client.get("/api/months?camera=cam1&year=2024")
    assert r.status_code == 200
    assert r.json() == ["01"]


def test_days(client):
    r = client.get("/api/days?camera=cam1&year=2024&month=01")
    assert r.status_code == 200
    assert r.json() == ["15"]


def test_images_desc(client):
    r = client.get("/api/images?camera=cam1&year=2024&month=01&day=15&sort=desc&page=1")
    assert r.status_code == 200
    data = r.json()
    assert data["total_pages"] == 1
    assert data["page"] == 1
    names = [img["filename"] for img in data["images"]]
    # desc order: newest first — 120000 before 110000 before startup_060000
    assert names[0] == "20240115_120000.jpg"
    assert names[-1] == "startup_20240115_060000.jpg"


def test_images_asc(client):
    r = client.get("/api/images?camera=cam1&year=2024&month=01&day=15&sort=asc&page=1")
    assert r.status_code == 200
    data = r.json()
    names = [img["filename"] for img in data["images"]]
    assert names[0] == "startup_20240115_060000.jpg"
    assert names[-1] == "20240115_120000.jpg"


def test_images_startup_badge(client):
    r = client.get("/api/images?camera=cam1&year=2024&month=01&day=15&sort=desc&page=1")
    data = r.json()
    startup = next(img for img in data["images"] if img["filename"].startswith("startup_"))
    assert startup["is_startup"] is True
    normal = next(img for img in data["images"] if not img["filename"].startswith("startup_"))
    assert normal["is_startup"] is False
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
pytest tests/website/test_api.py -v
```

Expected: `test_cameras` PASS, rest FAIL with 404 or attribute errors.

- [ ] **Step 3: Implement remaining endpoints in `website/main.py`**

Replace the full contents of `website/main.py` with:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/website/test_api.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add website/main.py tests/website/test_api.py
git commit -m "feat(website): add years/months/days/images API endpoints"
```

---

### Task 3: Image file serving and static mount

**Files:**
- Modify: `website/main.py`
- Modify: `tests/website/test_api.py`
- Create: `website/static/index.html` (placeholder for mount to work)
- Create: `website/static/app.js` (placeholder)
- Create: `website/static/style.css` (placeholder)

- [ ] **Step 1: Add test for image file serving**

Append to `tests/website/test_api.py`:

```python
def test_serve_image(client, image_root):
    r = client.get("/images/cam1/2024/01/15/20240115_120000.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


def test_serve_image_not_found(client):
    r = client.get("/images/cam1/2024/01/15/nope.jpg")
    assert r.status_code == 404
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/website/test_api.py::test_serve_image tests/website/test_api.py::test_serve_image_not_found -v
```

Expected: 404 or connection error — route doesn't exist yet.

- [ ] **Step 3: Add image serving route and static mount to `website/main.py`**

Add after `list_images` function and before the end of `main.py`:

```python
@app.get("/images/{camera}/{year}/{month}/{day}/{filename}")
def serve_image(camera: str, year: str, month: str, day: str, filename: str):
    path = IMAGE_ROOT / camera / year / month / day / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")


# Static files — mounted last so API routes take priority
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
```

- [ ] **Step 4: Create placeholder static files**

```bash
mkdir -p website/static
touch website/static/index.html website/static/app.js website/static/style.css
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/website/test_api.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add website/main.py website/static/
git commit -m "feat(website): add image serving route and static file mount"
```

---

### Task 4: HTML shell and CSS

**Files:**
- Modify: `website/static/index.html`
- Modify: `website/static/style.css`

No new tests — HTML/CSS are verified visually in Task 8.

- [ ] **Step 1: Write `website/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Car Counter Images</title>
  <link rel="stylesheet" href="/style.css" />
</head>
<body>
  <div id="app">
    <header>
      <div id="filters">
        <label>Camera
          <select id="sel-camera"></select>
        </label>
        <label>Year
          <select id="sel-year"></select>
        </label>
        <label>Month
          <select id="sel-month"></select>
        </label>
        <label>Day
          <select id="sel-day"></select>
        </label>
        <label>Sort
          <select id="sel-sort">
            <option value="desc" selected>Newest first</option>
            <option value="asc">Oldest first</option>
          </select>
        </label>
      </div>
    </header>

    <main>
      <div id="grid"></div>
    </main>

    <footer>
      <button id="btn-prev">Prev</button>
      <span id="page-indicator">Page 1 of 1</span>
      <button id="btn-next">Next</button>
    </footer>
  </div>

  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `website/static/style.css`**

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: system-ui, sans-serif;
  background: #111;
  color: #eee;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

header {
  padding: 12px 16px;
  background: #1e1e1e;
  border-bottom: 1px solid #333;
}

#filters {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
}

#filters label {
  display: flex;
  flex-direction: column;
  font-size: 0.75rem;
  color: #aaa;
  gap: 4px;
}

#filters select {
  background: #2a2a2a;
  color: #eee;
  border: 1px solid #444;
  border-radius: 4px;
  padding: 4px 8px;
  font-size: 0.9rem;
}

main {
  flex: 1;
  padding: 16px;
}

#grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
}

.thumb-card {
  background: #1e1e1e;
  border-radius: 6px;
  overflow: hidden;
  cursor: pointer;
  text-decoration: none;
  color: inherit;
  display: block;
}

.thumb-card img {
  width: 100%;
  height: 150px;
  object-fit: cover;
  display: block;
}

.thumb-info {
  padding: 6px 8px;
  font-size: 0.75rem;
  color: #bbb;
  display: flex;
  align-items: center;
  gap: 6px;
}

.badge-startup {
  background: #e67e22;
  color: #fff;
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 0.65rem;
  font-weight: bold;
  text-transform: uppercase;
}

footer {
  padding: 12px 16px;
  background: #1e1e1e;
  border-top: 1px solid #333;
  display: flex;
  align-items: center;
  gap: 16px;
  justify-content: center;
}

footer button {
  background: #2a2a2a;
  color: #eee;
  border: 1px solid #444;
  border-radius: 4px;
  padding: 6px 16px;
  cursor: pointer;
  font-size: 0.9rem;
}

footer button:disabled {
  opacity: 0.4;
  cursor: default;
}

#page-indicator {
  font-size: 0.85rem;
  color: #aaa;
}
```

- [ ] **Step 3: Commit**

```bash
git add website/static/index.html website/static/style.css
git commit -m "feat(website): HTML shell and CSS grid layout"
```

---

### Task 5: JavaScript — cascading dropdowns

**Files:**
- Modify: `website/static/app.js`

No automated tests — logic is exercised end-to-end in Task 8.

- [ ] **Step 1: Write the dropdown logic in `website/static/app.js`**

```javascript
const selCamera = document.getElementById('sel-camera');
const selYear   = document.getElementById('sel-year');
const selMonth  = document.getElementById('sel-month');
const selDay    = document.getElementById('sel-day');
const selSort   = document.getElementById('sel-sort');
const grid      = document.getElementById('grid');
const btnPrev   = document.getElementById('btn-prev');
const btnNext   = document.getElementById('btn-next');
const pageInd   = document.getElementById('page-indicator');

let currentPage = 1;

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${url}`);
  return r.json();
}

function populate(sel, values, selectedValue) {
  sel.innerHTML = '';
  values.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    if (v === selectedValue) opt.selected = true;
    sel.appendChild(opt);
  });
}

async function loadCameras() {
  const cameras = await fetchJSON('/api/cameras');
  if (!cameras.length) return;
  populate(selCamera, cameras, cameras[0]);
  await loadYears();
}

async function loadYears() {
  const cam = selCamera.value;
  const years = await fetchJSON(`/api/years?camera=${encodeURIComponent(cam)}`);
  // Most recent year = last in sorted list
  const latest = years[years.length - 1];
  populate(selYear, years, latest);
  await loadMonths();
}

async function loadMonths() {
  const cam  = selCamera.value;
  const year = selYear.value;
  const months = await fetchJSON(`/api/months?camera=${encodeURIComponent(cam)}&year=${encodeURIComponent(year)}`);
  const latest = months[months.length - 1];
  populate(selMonth, months, latest);
  await loadDays();
}

async function loadDays() {
  const cam   = selCamera.value;
  const year  = selYear.value;
  const month = selMonth.value;
  const days  = await fetchJSON(`/api/days?camera=${encodeURIComponent(cam)}&year=${encodeURIComponent(year)}&month=${encodeURIComponent(month)}`);
  const latest = days[days.length - 1];
  populate(selDay, days, latest);
  currentPage = 1;
  await loadImages();
}

// Event listeners — parent resets children
selCamera.addEventListener('change', loadYears);
selYear.addEventListener('change', loadMonths);
selMonth.addEventListener('change', loadDays);
selDay.addEventListener('change', () => { currentPage = 1; loadImages(); });
selSort.addEventListener('change', () => { currentPage = 1; loadImages(); });
```

- [ ] **Step 2: Commit**

```bash
git add website/static/app.js
git commit -m "feat(website): cascading dropdown logic"
```

---

### Task 6: JavaScript — image grid and pagination

**Files:**
- Modify: `website/static/app.js`

- [ ] **Step 1: Append image rendering and pagination to `website/static/app.js`**

```javascript
async function loadImages() {
  const cam   = selCamera.value;
  const year  = selYear.value;
  const month = selMonth.value;
  const day   = selDay.value;
  const sort  = selSort.value;

  const url = `/api/images?camera=${encodeURIComponent(cam)}&year=${encodeURIComponent(year)}&month=${encodeURIComponent(month)}&day=${encodeURIComponent(day)}&sort=${sort}&page=${currentPage}`;
  const data = await fetchJSON(url);

  grid.innerHTML = '';
  data.images.forEach(img => {
    const a = document.createElement('a');
    a.href = img.url;
    a.target = '_blank';
    a.className = 'thumb-card';

    const image = document.createElement('img');
    image.src = img.url;
    image.alt = img.timestamp;
    image.loading = 'lazy';

    const info = document.createElement('div');
    info.className = 'thumb-info';
    info.textContent = img.timestamp;

    if (img.is_startup) {
      const badge = document.createElement('span');
      badge.className = 'badge-startup';
      badge.textContent = 'startup';
      info.appendChild(badge);
    }

    a.appendChild(image);
    a.appendChild(info);
    grid.appendChild(a);
  });

  pageInd.textContent = `Page ${data.page} of ${data.total_pages}`;
  btnPrev.disabled = data.page <= 1;
  btnNext.disabled = data.page >= data.total_pages;
}

btnPrev.addEventListener('click', () => { currentPage--; loadImages(); });
btnNext.addEventListener('click', () => { currentPage++; loadImages(); });

// Bootstrap on page load
loadCameras();
```

- [ ] **Step 2: Commit**

```bash
git add website/static/app.js
git commit -m "feat(website): image grid and pagination"
```

---

### Task 7: Dockerfile and docker-compose

**Files:**
- Create: `website/Dockerfile`
- Create: `website/docker-compose.yml`

- [ ] **Step 1: Write `website/Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY static/ static/

ENV IMAGE_ROOT=/data

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 2: Write `website/docker-compose.yml`**

```yaml
services:
  website:
    build: .
    ports:
      - "8080:8080"
    environment:
      IMAGE_ROOT: /data
    volumes:
      - ../output:/data:ro
```

- [ ] **Step 3: Commit**

```bash
git add website/Dockerfile website/docker-compose.yml
git commit -m "feat(website): Dockerfile and docker-compose for local testing"
```

---

### Task 9: CI/CD — add website job to GitHub Actions

**Files:**
- Modify: `.github/workflows/docker-build-push.yml`

- [ ] **Step 1: Add website build job**

The existing file has one job `build-and-push`. Add a second job `build-and-push-website` inside the `jobs:` block. **Do NOT add a `needs:` key** — both jobs must run in parallel. Also verify the existing `build-and-push` job has no `concurrency:` group that would serialize runs; if it does, give the new job its own group name.

The new job is identical in structure to `build-and-push`, but differs in:
- `context:` is `website/`
- metadata `images:` and cache refs use `car-counter-website` instead of `car-counter`

Append inside the `jobs:` block of `.github/workflows/docker-build-push.yml`:

```yaml
  build-and-push-website:
    permissions:
      contents: read
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v6
      with:
        fetch-depth: 0

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v4

    - name: Log in to Docker Registry
      uses: docker/login-action@v4
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Generate semantic version
      id: version
      run: |
        LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
        LATEST_VERSION=${LATEST_TAG#v}
        echo "version=$LATEST_VERSION" >> $GITHUB_OUTPUT

    - name: Extract metadata for Docker
      id: meta
      uses: docker/metadata-action@v6
      with:
        images: ${{ env.REGISTRY }}/car-counter-website
        tags: |
          type=ref,event=branch
          type=semver,pattern={{version}},value=${{ steps.version.outputs.version }}
          type=semver,pattern={{major}}.{{minor}},value=${{ steps.version.outputs.version }}
          type=semver,pattern={{major}},value=${{ steps.version.outputs.version }}
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push website Docker image
      uses: docker/build-push-action@v7
      with:
        context: website/
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        platforms: linux/amd64,linux/arm64
        cache-from: |
          type=registry,ref=${{ env.REGISTRY }}/car-counter-website:latest
          type=gha,scope=car-counter-website
        cache-to: type=gha,scope=car-counter-website,mode=max
```

> Note: add `scope=car-counter-website` to the website job's GHA cache to avoid collisions with the main image cache.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/docker-build-push.yml
git commit -m "ci: add website Docker build-and-push job"
```

---

### Task 10: Update kubernetes-deployment.md

**Files:**
- Modify: `docs/kubernetes-deployment.md`

The Kubernetes manifests live **only** in this doc — there is no `website/k8s/` folder.

- [ ] **Step 1: Append website deployment section**

Add the following at the end of `docs/kubernetes-deployment.md`:

````markdown
---

## Website (Image Browser)

A single-replica web UI that serves annotated images from the shared PVC.

### Manifests

**deployment.yaml**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: car-counter-website
  namespace: car-counter
spec:
  replicas: 1
  selector:
    matchLabels:
      app: car-counter-website
  template:
    metadata:
      labels:
        app: car-counter-website
    spec:
      containers:
        - name: website
          image: docker.thehormanns.net/car-counter-website:latest
          env:
            - name: IMAGE_ROOT
              value: /data
          ports:
            - containerPort: 8080
              name: http
          volumeMounts:
            - name: images
              mountPath: /data
              readOnly: true
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 256Mi
      volumes:
        - name: images
          persistentVolumeClaim:
            claimName: car-counter-output
```

**service.yaml**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: car-counter-website
  namespace: car-counter
  annotations:
    metallb.universe.tf/address-pool: first-ip-pool
spec:
  type: LoadBalancer
  selector:
    app: car-counter-website
  ports:
    - port: 80
      targetPort: 8080
      protocol: TCP
```

### Deploy

```bash
kubectl apply -f - <<'EOF'
# paste deployment.yaml contents here
EOF

kubectl apply -f - <<'EOF'
# paste service.yaml contents here
EOF
```

Or save each block above to a temp file and `kubectl apply -f <file>`.

The pod mounts the same `car-counter-output` PVC **read-only** at `/data`.

### Get the external IP

```bash
kubectl get service car-counter-website -n car-counter
```

The `EXTERNAL-IP` column shows the MetalLB address. Browse to `http://<EXTERNAL-IP>/`.

### Updating the image

```bash
kubectl rollout restart deployment/car-counter-website -n car-counter
```
````

- [ ] **Step 2: Commit**

```bash
git add docs/kubernetes-deployment.md
git commit -m "docs: add website Kubernetes manifests and deployment steps"
```

---

### Task 11: Local smoke test

This task has no code commits — it verifies the full stack works locally before closing the branch.

- [ ] **Step 1: Ensure `../output` has at least one image subfolder**

The `docker-compose.yml` mounts `../output` (i.e., `/home/ghormann/src/car-counter/output`). If that folder is empty, create a minimal test image:

```bash
mkdir -p output/testcam/2024/01/15
cp /dev/null output/testcam/2024/01/15/20240115_120000.jpg
```

- [ ] **Step 2: Build and run with docker-compose**

```bash
cd website
docker compose up --build
```

Expected: container starts, logs show `Uvicorn running on http://0.0.0.0:8080`.

- [ ] **Step 3: Verify in browser**

Open `http://localhost:8080/` — confirm:
- Camera dropdown auto-selects `testcam`
- Year/month/day cascade correctly
- Thumbnail grid renders (even a broken image icon counts — the file is empty)
- Prev/Next buttons disable correctly on a single-page result

- [ ] **Step 4: Stop docker-compose**

```bash
docker compose down
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|---|---|
| FastAPI serves API + static, port 8080 | Task 1 + Task 3 + Task 7 |
| `IMAGE_ROOT` env var, default `/data` | Task 1 |
| Cascading dropdowns, auto-select on load | Task 5 |
| No "All" option | Task 5 (populate never adds an All option) |
| Sort toggle newest/oldest | Task 5 + Task 6 |
| Default sort: newest first | Task 5 (selSort default value="desc") |
| Thumbnails full-size scaled by browser | Task 6 |
| Max 50 per page | Task 2 (PAGE_SIZE = 50) |
| Human-readable timestamp under each thumb | Task 6 |
| Startup badge | Task 2 (`is_startup`) + Task 6 (badge render) |
| Click thumbnail → new tab | Task 6 (`target='_blank'`) |
| Prev/Next + Page X of Y | Task 6 |
| All 7 API endpoints | Task 1 + Task 2 + Task 3 |
| `GET /` serves the SPA | Task 3 (StaticFiles html=True) |
| Dockerfile, port 8080 | Task 7 |
| docker-compose.yml for local dev | Task 7 |
| K8s Deployment + Service (MetalLB) | Task 10 (inline in kubernetes-deployment.md) |
| PVC read-only at `/data` | Task 10 |
| Resource limits (100m/128Mi → 500m/256Mi) | Task 10 |
| Same namespace as car-counter | Task 10 |
| CI: website image in GHA workflow | Task 9 |
| Registry: `docker.thehormanns.net/car-counter-website` | Task 9 |
| Same platforms and versioning | Task 9 |
| GHA cache strategy shared | Task 9 (scoped to avoid collision) |
| Update `kubernetes-deployment.md` | Task 10 |

All requirements covered. No TBDs or placeholders found.
