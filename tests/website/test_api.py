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
