import json
import pytest
import yaml
from pathlib import Path

from src.config import load_app_config, load_mqtt_config, AppConfig, MqttConfig, ScanRegion


VALID_APP_DATA = {
    'camera_name': 'driveway',
    'rtsps_url': 'rtsps://192.168.1.10/stream',
    'vehicle_classes': ['car', 'truck', 'bus'],
    'detection_confidence': 0.4,
    'stationary_seconds': 3,
    'iou_threshold': 0.5,
    'night_enhancement': True,
    'night_brightness_threshold': 80,
    'ir_saturation_threshold': 30,
    'target_fps': 1,
    'model_path': 'yolov8n.pt',
    'publish_interval_seconds': 5,
    'mqtt_timeout_seconds': 60,
    'mqtt_topic': 'car-counter/driveway',
    'image_save_cooldown_seconds': 30,
    # output_dir is set per-test using tmp_path
}

VALID_MQTT_DATA = {
    'host': 'server.com',
    'port': 1883,
    'username': 'user',
    'password': 'pass',
    'extra_ignored_field': 'ignored',
}


def write_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f)


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def make_app_config_file(tmp_path):
    data = dict(VALID_APP_DATA)
    data['output_dir'] = str(tmp_path)
    p = tmp_path / 'config.yaml'
    write_yaml(p, data)
    return p


class TestLoadAppConfig:
    def test_valid_config_returns_app_config(self, tmp_path):
        p = make_app_config_file(tmp_path)
        result = load_app_config(str(p))
        assert isinstance(result, AppConfig)
        assert result.camera_name == 'driveway'
        assert result.rtsps_url == 'rtsps://192.168.1.10/stream'
        assert result.detection_confidence == 0.4
        assert result.output_dir == tmp_path

    def test_scan_regions_parsed_when_present(self, tmp_path):
        data = dict(VALID_APP_DATA)
        data['output_dir'] = str(tmp_path)
        data['scan_regions'] = [{'x': 100, 'y': 200, 'width': 400, 'height': 300}]
        p = tmp_path / 'config.yaml'
        write_yaml(p, data)
        result = load_app_config(str(p))
        assert result.scan_regions == [ScanRegion(x=100, y=200, width=400, height=300)]

    def test_scan_regions_defaults_to_empty_list(self, tmp_path):
        p = make_app_config_file(tmp_path)
        result = load_app_config(str(p))
        assert result.scan_regions == []

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_app_config(str(tmp_path / 'nonexistent.yaml'))

    def test_invalid_yaml_raises_value_error(self, tmp_path):
        p = tmp_path / 'config.yaml'
        p.write_text("{{bad: yaml: :")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_app_config(str(p))

    @pytest.mark.parametrize("missing_field", [
        'camera_name', 'rtsps_url', 'vehicle_classes', 'detection_confidence',
        'stationary_seconds', 'iou_threshold', 'target_fps', 'model_path',
        'publish_interval_seconds', 'mqtt_timeout_seconds', 'mqtt_topic',
        'output_dir', 'image_save_cooldown_seconds',
    ])
    def test_missing_required_field_raises_value_error(self, tmp_path, missing_field):
        data = dict(VALID_APP_DATA)
        data['output_dir'] = str(tmp_path)
        del data[missing_field]
        p = tmp_path / 'config.yaml'
        write_yaml(p, data)
        with pytest.raises(ValueError, match=f"'{missing_field}'"):
            load_app_config(str(p))

    def test_rtsps_url_without_rtsps_scheme_raises(self, tmp_path):
        data = dict(VALID_APP_DATA)
        data['output_dir'] = str(tmp_path)
        data['rtsps_url'] = 'rtsp://192.168.1.10/stream'
        p = tmp_path / 'config.yaml'
        write_yaml(p, data)
        with pytest.raises(ValueError, match="rtsps://"):
            load_app_config(str(p))

    def test_nonexistent_output_dir_raises(self, tmp_path):
        data = dict(VALID_APP_DATA)
        data['output_dir'] = str(tmp_path / 'does_not_exist')
        p = tmp_path / 'config.yaml'
        write_yaml(p, data)
        with pytest.raises(ValueError, match="output_dir"):
            load_app_config(str(p))

    def test_output_dir_tilde_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv('HOME', str(tmp_path))
        (tmp_path / 'myoutput').mkdir()
        data = dict(VALID_APP_DATA)
        data['output_dir'] = '~/myoutput'
        p = tmp_path / 'config.yaml'
        write_yaml(p, data)
        result = load_app_config(str(p))
        assert result.output_dir == tmp_path / 'myoutput'


class TestLoadMqttConfig:
    def test_valid_config_returns_mqtt_config(self, tmp_path):
        p = tmp_path / 'mqtt.json'
        write_json(p, VALID_MQTT_DATA)
        result = load_mqtt_config(str(p))
        assert isinstance(result, MqttConfig)
        assert result.host == 'server.com'
        assert result.port == 1883
        assert result.username == 'user'
        assert result.password == 'pass'

    def test_extra_fields_are_ignored(self, tmp_path):
        p = tmp_path / 'mqtt.json'
        write_json(p, VALID_MQTT_DATA)
        result = load_mqtt_config(str(p))
        assert not hasattr(result, 'extra_ignored_field')

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_mqtt_config(str(tmp_path / 'nonexistent.json'))

    def test_invalid_json_raises_value_error(self, tmp_path):
        p = tmp_path / 'mqtt.json'
        p.write_text("{invalid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_mqtt_config(str(p))

    @pytest.mark.parametrize("missing_field", ['host', 'port', 'username', 'password'])
    def test_missing_required_field_raises_value_error(self, tmp_path, missing_field):
        data = dict(VALID_MQTT_DATA)
        del data[missing_field]
        p = tmp_path / 'mqtt.json'
        write_json(p, data)
        with pytest.raises(ValueError, match=f"'{missing_field}'"):
            load_mqtt_config(str(p))
