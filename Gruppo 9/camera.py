"""Acquisizione webcam e inferenza MediaPipe in un unico thread condiviso."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import cv2

# Nasconde i messaggi INFO/WARNING interni di MediaPipe; gli errori restano visibili.
os.environ.setdefault("GLOG_minloglevel", "2")

import mediapipe as mp
import numpy as np

from camera_devices import resolve_camera_index
from hand_tracking import draw_hand_landmarks, draw_status, normalized_pinch_distance
from measurements import MeasurementBuffer


class CameraService:
    """Possiede l'unico VideoCapture e pubblica l'ultimo JPEG ai client Flask."""

    def __init__(
        self,
        *,
        model_path: Path,
        camera_index: int | None,
        frame_width: int,
        frame_height: int,
        jpeg_quality: int,
        mirror: bool,
        measurements: MeasurementBuffer,
    ) -> None:
        self.model_path = model_path
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.jpeg_quality = jpeg_quality
        self.mirror = mirror
        self.measurements = measurements

        self._condition = threading.Condition()
        self._start_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_jpeg: bytes | None = None
        self._frame_sequence = 0
        self._error: str | None = None
        self._active_camera_index: int | None = None
        self._camera_name: str | None = None

    @property
    def status(self) -> dict[str, str | bool | int | None]:
        with self._condition:
            return {
                "running": self._thread is not None and self._thread.is_alive(),
                "error": self._error,
                "camera_index": self._active_camera_index,
                "camera_name": self._camera_name,
            }

    def start(self, *, permission_wait_seconds: float = 0.0) -> None:
        """Apre la webcam e avvia una sola volta il worker condiviso.

        La prima chiamata avviene dal main thread, requisito importante perche
        AVFoundation possa mostrare la richiesta permessi di macOS.
        """

        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            with self._condition:
                self._error = None

            capture = self._open_camera()
            if not capture.isOpened() and permission_wait_seconds > 0:
                capture.release()
                print(
                    "In attesa del permesso Fotocamera di macOS. "
                    "Se compare la finestra, scegli Consenti..."
                )
                deadline = time.monotonic() + permission_wait_seconds
                while time.monotonic() < deadline and not self._stop_event.is_set():
                    time.sleep(0.75)
                    capture = self._open_camera()
                    if capture.isOpened():
                        break
                    capture.release()

            if not capture.isOpened():
                message = (
                    f"Impossibile aprire la webcam con indice "
                    f"{self._active_camera_index}. Controlla Impostazioni di Sistema "
                    "> Privacy e sicurezza > Fotocamera, poi riavvia l'app."
                )
                self._set_error(message)
                self._publish_error_frame(message)
                return

            self._thread = threading.Thread(
                target=self._capture_loop,
                args=(capture,),
                name="camera-mediapipe",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)

    def mjpeg_stream(self) -> Iterator[bytes]:
        """Consegna ogni nuovo frame senza eseguire una seconda acquisizione."""

        self.start()
        last_sequence = -1

        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait_for(
                    lambda: self._frame_sequence != last_sequence
                    or self._stop_event.is_set(),
                    timeout=2.0,
                )
                if self._stop_event.is_set():
                    return
                jpeg = self._latest_jpeg
                last_sequence = self._frame_sequence

            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )

    def _capture_loop(self, capture: cv2.VideoCapture) -> None:
        try:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(self.model_path),
                    delegate=mp.tasks.BaseOptions.Delegate.CPU,
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
                last_media_timestamp_ms = -1

                while not self._stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError("La webcam non ha restituito un frame.")
                    if self.mirror:
                        frame = cv2.flip(frame, 1)

                    media_timestamp_ms = time.monotonic_ns() // 1_000_000
                    media_timestamp_ms = max(
                        media_timestamp_ms, last_media_timestamp_ms + 1
                    )
                    last_media_timestamp_ms = media_timestamp_ms

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(rgb_frame),
                    )
                    result = landmarker.detect_for_video(
                        mp_image, media_timestamp_ms
                    )

                    distance = None
                    if result.hand_landmarks:
                        landmarks = result.hand_landmarks[0]
                        distance = normalized_pinch_distance(landmarks)
                        draw_hand_landmarks(frame, landmarks)

                    draw_status(frame, distance)
                    wall_timestamp_ms = time.time_ns() // 1_000_000
                    self.measurements.append(wall_timestamp_ms, distance)
                    self._publish_frame(frame)

        except Exception as exc:  # L'errore viene mostrato anche nello stream web.
            self._set_error(str(exc))
            self._publish_error_frame(str(exc))
        finally:
            capture.release()
            with self._condition:
                self._condition.notify_all()

    def _open_camera(self) -> cv2.VideoCapture:
        camera_index, camera_name = resolve_camera_index(self.camera_index)
        with self._condition:
            self._active_camera_index = camera_index
            self._camera_name = camera_name

        selected = camera_name or f"indice {camera_index}"
        print(f"Webcam selezionata: {selected} (OpenCV index {camera_index})")

        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            return cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        return cv2.VideoCapture(camera_index)

    def _publish_frame(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return

        with self._condition:
            self._latest_jpeg = encoded.tobytes()
            self._frame_sequence += 1
            self._condition.notify_all()

    def _publish_error_frame(self, message: str) -> None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Errore webcam / MediaPipe",
            (35, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (80, 80, 255),
            2,
            cv2.LINE_AA,
        )
        short_message = message if len(message) <= 70 else f"{message[:67]}..."
        cv2.putText(
            frame,
            short_message,
            (35, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        self._publish_frame(frame)

    def _set_error(self, message: str) -> None:
        with self._condition:
            self._error = message
