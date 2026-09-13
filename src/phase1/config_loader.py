"""Configuration loader for Phase 1 pipeline."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Supports environment variable overrides via pattern: PHASE1_<KEY>
    
    Args:
        config_path: Path to config.yaml. If None, looks for config.yaml in src/phase1/
    
    Returns:
        Dictionary with all configuration parameters
    
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If required keys are missing
    """
    # Determine config path
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    # Validate required keys
    required_keys = [
        "UBER_HANDLES",
        "MAX_TURNS",
        "ENFORCE_ALTERNATION",
        "POSITIVE_CLOSERS",
        "BOILERPLATE_MIN_OCCURRENCES",
        "BOILERPLATE_MIN_SHARE",
        "LANGUAGE_FILTERING",
        "ACCEPTED_LANGUAGES",
        "INPUT_CSV",
        "OUTPUT_DIR",
        "OUTPUT_PARQUET",
        "LOG_LEVEL",
        "LOG_FILE"
    ]
    
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    # Apply environment variable overrides
    for key in config.keys():
        env_key = f"PHASE1_{key}"
        if env_key in os.environ:
            config[key] = os.environ[env_key]
    
    return config


if __name__ == "__main__":
    config = load_config()
    print("Configuration loaded successfully:")
    for key, value in config.items():
        print(f"  {key}: {value}")
