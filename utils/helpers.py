"""
Utility helpers for file system, image handling, and JSON schema validation.
"""
import base64
import json
import io
import re
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GENERATIONS_DIR = OUTPUTS_DIR / "generations"
OUTLINES_DIR = OUTPUTS_DIR / "outlines"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
DATASET_DIR = PROJECT_ROOT / "dataset"

for d in [OUTPUTS_DIR, GENERATIONS_DIR, OUTLINES_DIR, EXPERIMENTS_DIR, DATASET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def base64_to_image(b64_string: str) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def save_image_to_disk(image: Image.Image, path: Union[str, Path]) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(target_path))
    return target_path

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name.lower()).strip('_')
