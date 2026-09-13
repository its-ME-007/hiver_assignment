"""Unit tests for configuration loading."""
import pytest
import os
from pathlib import Path
from src.phase1.config_loader import load_config


def test_load_config_success():
    """Test successful config loading."""
    config_path = Path(__file__).parent.parent / "src" / "phase1" / "config.yaml"
    config = load_config(str(config_path))
    
    assert isinstance(config, dict)
    assert len(config) > 0
    assert "UBER_HANDLES" in config
    assert isinstance(config["UBER_HANDLES"], list)


def test_load_config_required_keys():
    """Test that all required keys are present."""
    config_path = Path(__file__).parent.parent / "src" / "phase1" / "config.yaml"
    config = load_config(str(config_path))
    
    required_keys = [
        "UBER_HANDLES",
        "MAX_TURNS",
        "ENFORCE_ALTERNATION",
        "POSITIVE_CLOSERS",
        "BOILERPLATE_MIN_OCCURRENCES",
        "BOILERPLATE_MIN_SHARE",
    ]
    
    for key in required_keys:
        assert key in config, f"Missing required key: {key}"


def test_load_config_missing_file():
    """Test error handling for missing config file."""
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_load_config_values():
    """Test that config values have expected types."""
    config_path = Path(__file__).parent.parent / "src" / "phase1" / "config.yaml"
    config = load_config(str(config_path))
    
    assert isinstance(config["MAX_TURNS"], int)
    assert isinstance(config["ENFORCE_ALTERNATION"], bool)
    assert isinstance(config["BOILERPLATE_MIN_OCCURRENCES"], int)
    assert isinstance(config["BOILERPLATE_MIN_SHARE"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
