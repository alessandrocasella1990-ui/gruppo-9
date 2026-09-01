"""Selezione della webcam integrata su macOS."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


def resolve_camera_index(configured_index: int | None) -> tuple[int, str | None]:
    """Restituisce l'indice OpenCV, preferendo la camera integrata del Mac.

    OpenCV/AVFoundation ordina i dispositivi per uniqueID. Ricostruiamo lo stesso
    ordinamento dai dati di system_profiler, evitando che Continuity Camera venga
    scelta solo perche il suo ID corrisponde all'indice 0.
    """

    if configured_index is not None:
        return configured_index, None
    if sys.platform != "darwin":
        return 0, None

    devices = _mac_camera_devices()
    if not devices:
        return 0, None

    devices.sort(key=lambda device: str(device.get("spcamera_unique-id", "")))

    for index, device in enumerate(devices):
        name = str(device.get("_name", ""))
        model = str(device.get("spcamera_model-id", ""))
        label = f"{name} {model}".replace("\u00a0", " ").casefold()

        is_continuity_camera = any(
            token in label
            for token in ("iphone", "ipad", "continuity", "vista scrivania")
        )
        is_builtin_camera = any(
            token in label
            for token in ("macbook", "facetime", "built-in", "integrata")
        )
        if is_builtin_camera and not is_continuity_camera:
            return index, name

    first_name = str(devices[0].get("_name", "")) or None
    return 0, first_name


def _mac_camera_devices() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if completed.returncode != 0:
            return []
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []

    devices = payload.get("SPCameraDataType", [])
    return [device for device in devices if isinstance(device, dict)]
