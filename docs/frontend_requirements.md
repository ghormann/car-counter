# Frontend Requirements

## Purpose

A lightweight debug tool for browsing annotated images saved to the k8s PVC by the car-counter service.

---

## Tech Stack

- **Backend**: Python + FastAPI (single process, serves both API and static files)
- **Frontend**: Plain HTML/JS/CSS — no framework, no build step
- **Image serving**: FastAPI streams full-size images directly from the mounted PVC (no thumbnail generation)
- **Port**: Container listens on **8080**

---

## Image Storage Layout

Images are stored on the PVC at:

```
{IMAGE_ROOT}/{camera}/{year}/{month}/{day}/{timestamp}.jpg
```

- `IMAGE_ROOT` is configurable via environment variable, default `/data`
- Startup images are prefixed: `startup_{timestamp}.jpg`
- Timestamp format in filenames: `YYYYMMDD_HHMMSS`

---

## UI Layout

Single-page layout:

1. **Filter + sort bar** across the top
2. **Thumbnail grid** below
3. **Pagination controls** at the bottom

---

## Filtering

- **Cascading dropdowns**: Camera → Year → Month → Day
- Each dropdown is populated from folders that actually exist on disk
- No "All" option — each level requires a selection
- On page load: auto-select the first camera alphabetically, then cascade to the most recent year → month → day for that camera
- Changing a parent dropdown resets and repopulates all child dropdowns

---

## Sort Order

- Toggle near the filter bar: **Newest first / Oldest first**
- Default: **Newest first**
- Sorts by the timestamp embedded in the filename

---

## Thumbnail Display

- Thumbnails are full-size images scaled down by the browser (no server-side resizing)
- Grid layout, responsive
- Max **50 images per page**
- Under each thumbnail:
  - Human-readable timestamp: `YYYY-MM-DD HH:MM:SS`
  - "startup" badge if the filename begins with `startup_`
- Clicking a thumbnail opens an **in-page image viewer** (same tab) that shows the full-size image with a thin top bar containing back (←), prev (‹), and next (›) icons plus the image timestamp and startup badge; the image fills the remaining viewport height with `object-fit: contain` on a black background
- The viewer navigates within the current page's image list (up to 50 images); keyboard left/right arrows also navigate
- The back arrow returns to the grid with all filter and pagination state preserved
- Pagination: **Prev / Next** buttons with a **"Page X of Y"** indicator

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/cameras` | List available camera folders |
| `GET /api/years?camera=X` | List available years for a camera |
| `GET /api/months?camera=X&year=Y` | List available months |
| `GET /api/days?camera=X&year=Y&month=M` | List available days |
| `GET /api/images?camera=X&year=Y&month=M&day=D&sort=desc&page=1` | Paginated image list (max 50 per page) |
| `GET /images/{camera}/{year}/{month}/{day}/{filename}` | Serve a full-size image file |
| `GET /` | Serve the single-page HTML app |

---

## Authentication

None — internal debug tool, access controlled at the network level.

---

## Project Structure

```
website/
├── Dockerfile
├── main.py               # FastAPI app
├── requirements.txt
├── static/
│   ├── index.html
│   ├── app.js
│   └── style.css
└── k8s/
    ├── deployment.yaml
    └── service.yaml
```

---

## Local Testing

A `docker-compose.yml` inside `website/` should allow local testing by mounting a local image folder as `/data`. Example:

```yaml
services:
  website:
    build: .
    ports:
      - "8080:8080"
    environment:
      IMAGE_ROOT: /data
    volumes:
      - ../output:/data:ro  # point to local car-counter output folder
```

---

## CI/CD

The existing `.github/workflows/docker-build-push.yml` workflow should be updated to also build and push the website container on release. The website image should:

- Use `website/` as the Docker build context
- Be pushed to the same registry (`docker.thehormanns.net`) as `car-counter-website`
- Use the same semantic versioning tags and platforms (`linux/amd64`, `linux/arm64`)
- Share the same GHA cache and registry cache strategy as the car-counter job

---

## Kubernetes Deployment

We should modify the existing kuberneties-deployment.md file to provide steps for deploying this pod as well: 

- **Deployment**: single replica, mounts the car-counter PVC **read-only** at `/data`
- **Service**: `LoadBalancer` type using MetalLB (`metallb.universe.tf/address-pool: first-ip-pool`), port 80 → container port 8080
- Resource limits:
  ```yaml
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 256Mi
  ```
- Namespace: same as the car-counter deployment

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `IMAGE_ROOT` | `/data` | Root path where camera image folders are mounted |
