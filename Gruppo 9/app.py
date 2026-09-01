"""Entry point Flask: eseguire con `python app.py`."""

from __future__ import annotations

import atexit

from flask import Flask, Response, jsonify, render_template

from camera import CameraService
from config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    HOST,
    JPEG_QUALITY,
    MIRROR_CAMERA,
    MODEL_PATH,
    PORT,
    SERIES_MAX_LENGTH,
)
from measurements import MeasurementBuffer


measurements = MeasurementBuffer(SERIES_MAX_LENGTH)
camera = CameraService(
    model_path=MODEL_PATH,
    camera_index=CAMERA_INDEX,
    frame_width=FRAME_WIDTH,
    frame_height=FRAME_HEIGHT,
    jpeg_quality=JPEG_QUALITY,
    mirror=MIRROR_CAMERA,
    measurements=measurements,
)

app = Flask(__name__)


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/video_feed")
def video_feed() -> Response:
    return Response(
        camera.mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/api/latest")
def api_latest() -> Response:
    return jsonify(measurements.latest())


@app.get("/api/series")
def api_series() -> Response:
    samples = measurements.series()
    return jsonify({"count": len(samples), "samples": samples})


@app.get("/api/status")
def api_status() -> Response:
    return jsonify(camera.status)


def main() -> None:
    if not MODEL_PATH.is_file():
        raise SystemExit(
            f"Modello MediaPipe non trovato: {MODEL_PATH}\n"
            "Inserire hand_landmarker.task nella cartella models/."
        )

    camera.start(permission_wait_seconds=20.0)
    atexit.register(camera.stop)
    print(f"Apri il browser su http://{HOST}:{PORT}")
    try:
        app.run(
            host=HOST,
            port=PORT,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
