import pytest
from unittest.mock import patch
from src.__main__ import _utcnow


class TestUtcNow:
    def test_returns_iso8601_utc_string(self):
        result = _utcnow()
        # Format: 2026-06-13T14:32:01Z
        assert len(result) == 20
        assert result.endswith('Z')
        assert result[10] == 'T'
        assert result[4] == '-'
        assert result[7] == '-'
        assert result[13] == ':'
        assert result[16] == ':'

    def test_startup_exits_on_missing_config(self, tmp_path):
        with patch.dict('os.environ', {
            'APP_CONFIG_PATH': str(tmp_path / 'nonexistent.yaml'),
            'MQTT_CONFIG_PATH': str(tmp_path / 'nonexistent.json'),
        }):
            from src.__main__ import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code != 0
