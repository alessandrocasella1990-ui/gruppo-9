"""Configurazione centrale del progetto."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.environ.get("HAND_MODEL_PATH", BASE_DIR / "models" / "hand_landmarker.task")
).expanduser()

_camera_index = os.environ.get("CAMERA_INDEX")
CAMERA_INDEX = int(_camera_index) if _camera_index is not None else None
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "720"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "85"))
MIRROR_CAMERA = os.environ.get("MIRROR_CAMERA", "1").strip().casefold() not in {
    "0",
    "false",
    "no",
    "off",
}

# Circa 30 secondi a 20 FPS. deque elimina automaticamente i campioni piu vecchi.
SERIES_MAX_LENGTH = int(os.environ.get("SERIES_MAX_LENGTH", "600"))

HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
PORT = int(os.environ.get("FLASK_PORT", "8000"))
