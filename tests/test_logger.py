"""Unit tests for logging system."""
import pytest
import tempfile
from pathlib import Path
from src.phase1.logger import StructuredLogger, Metrics, setup_logger


def test_structured_logger_creation():
    """Test logger creation."""
    logger = StructuredLogger("test_logger")
    assert logger is not None


def test_structured_logger_with_file():
    """Test logger with file output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = StructuredLogger("test_file_logger_unique", log_file=str(log_file), log_level="INFO")
        
        logger.info("Test message")
        assert log_file.exists()
        
        # Clean up handlers
        for handler in logger.logger.handlers[:]:
            handler.close()
            logger.logger.removeHandler(handler)


def test_structured_logger_levels():
    """Test different log levels."""
    logger = StructuredLogger("test_logger", log_level="DEBUG")
    
    # Should not raise exceptions
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")


def test_metrics_operations():
    """Test metrics tracking."""
    metrics = Metrics()
    
    metrics.increment("test_counter")
    assert metrics.get("test_counter") == 1
    
    metrics.increment("test_counter", 5)
    assert metrics.get("test_counter") == 6
    
    metrics.set("test_value", 42)
    assert metrics.get("test_value") == 42


def test_metrics_to_dict():
    """Test metrics dictionary conversion."""
    metrics = Metrics()
    metrics.set("key1", "value1")
    metrics.set("key2", 42)
    
    metrics_dict = metrics.to_dict()
    assert isinstance(metrics_dict, dict)
    assert metrics_dict["key1"] == "value1"
    assert metrics_dict["key2"] == 42


def test_metrics_report():
    """Test metrics report generation."""
    metrics = Metrics()
    metrics.set("counter", 100)
    metrics.set("rate", 0.95)
    
    report = metrics.report()
    assert "counter" in report
    assert "100" in report


def test_setup_logger_from_config():
    """Test logger setup from config."""
    from src.phase1.config_loader import load_config
    config_path = Path(__file__).parent.parent / "src" / "phase1" / "config.yaml"
    config = load_config(str(config_path))
    
    logger = setup_logger(config)
    assert logger is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
