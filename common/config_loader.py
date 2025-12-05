import yaml
from pathlib import Path

_config_cache = None  # Cache configuration file

def load_config():  # Load config.yaml from project root directory, cache in memory for global reuse, avoid repeated file reads
    global _config_cache
    if _config_cache is None:  # If cache is empty, load configuration file
        # Project root directory
        project_root = Path(__file__).resolve().parents[1]
        config_path = project_root / 'config.yaml'
        if not config_path.exists():  # If configuration file does not exist, raise exception
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r', encoding="utf-8") as f:  # If exists, open configuration file and load into cache
            _config_cache = yaml.safe_load(f)
    return _config_cache


if __name__ == '__main__':  # Test
    print(load_config())  # Print configuration file

"""
Example output:

{'yolo': {'model_path': 'models/yolo/yolov8_best.pt'}, 'duckdb': {'path': 'duckdb/demo.db'}}
"""
