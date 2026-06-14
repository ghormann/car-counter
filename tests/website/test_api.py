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
