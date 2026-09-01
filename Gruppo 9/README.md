# Hand Landmarker realtime

Piccolo progetto locale: webcam -> MediaPipe Hand Landmarker -> buffer temporale -> Flask.

La webcam viene aperta una sola volta da `CameraService`. Il thread Flask espone lo
stesso frame JPEG a tutti i client con uno stream MJPEG. Su macOS il programma
preferisce automaticamente la camera integrata del Mac ed evita Continuity Camera
dell'iPhone; `CAMERA_INDEX` permette comunque di forzare un indice. Il video viene
mostrato a specchio e il pannello web disegna in tempo reale gli ultimi 300 campioni.

## Avvio su macOS Apple Silicon

È consigliato Python 3.11 o 3.12 installato nativamente per arm64.

```bash
cd "/Users/salvatorebosco/Desktop/Gruppo 9"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Aprire <http://127.0.0.1:8000>. Al primo avvio macOS potrebbe chiedere il permesso
di usare la fotocamera per Terminal (o per l'app da cui si esegue Python).

## Endpoint

- `/`: interfaccia web e video realtime
- `/video_feed`: stream MJPEG
- `/api/latest`: ultimo campione
- `/api/series`: buffer temporale (massimo 600 campioni per default)
- `/api/status`: stato del worker webcam

I timestamp delle API sono Unix epoch in millisecondi. La distanza è il rapporto
tra la distanza 2D pollice-indice (landmark 4-8) e la scala del palmo (0-9). Se non
c'è una mano nel frame, `distance_normalized` vale `null`.

Le impostazioni principali possono essere cambiate con variabili d'ambiente, per
esempio `CAMERA_INDEX=1`, `FLASK_PORT=8080` o `SERIES_MAX_LENGTH=1200`.
Impostare `MIRROR_CAMERA=0` per disattivare la visualizzazione a specchio.
