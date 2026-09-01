"""Calcolo della misura e disegno dei 21 landmark della mano."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import cv2


THUMB_TIP = 4
INDEX_FINGER_TIP = 8
WRIST = 0
MIDDLE_FINGER_MCP = 9

# Connessioni anatomiche tra i 21 landmark di MediaPipe.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),
)


def normalized_pinch_distance(landmarks: Sequence[Any]) -> float | None:
    """Distanza 4-8 divisa per la scala del palmo 0-9.

    Le coordinate normalizzate di MediaPipe rendono la misura indipendente dalla
    risoluzione del frame. Il rapporto con 0-9 riduce inoltre la dipendenza dalla
    distanza della mano dalla webcam.
    """

    if len(landmarks) < 21:
        return None

    thumb = landmarks[THUMB_TIP]
    index = landmarks[INDEX_FINGER_TIP]
    wrist = landmarks[WRIST]
    middle_mcp = landmarks[MIDDLE_FINGER_MCP]

    pinch_distance = math.hypot(thumb.x - index.x, thumb.y - index.y)
    hand_scale = math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)

    if hand_scale < 1e-6:
        return None
    return pinch_distance / hand_scale


def draw_hand_landmarks(frame: Any, landmarks: Sequence[Any]) -> None:
    """Disegna connessioni e punti sul frame BGR in-place."""

    height, width = frame.shape[:2]
    points = [
        (
            min(max(int(landmark.x * width), 0), width - 1),
            min(max(int(landmark.y * height), 0), height - 1),
        )
        for landmark in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (70, 220, 120), 2, cv2.LINE_AA)

    for index, point in enumerate(points):
        color = (50, 80, 255) if index in (THUMB_TIP, INDEX_FINGER_TIP) else (0, 215, 255)
        radius = 6 if index in (THUMB_TIP, INDEX_FINGER_TIP) else 4
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)

    cv2.line(
        frame,
        points[THUMB_TIP],
        points[INDEX_FINGER_TIP],
        (255, 100, 50),
        3,
        cv2.LINE_AA,
    )


def draw_status(frame: Any, distance: float | None) -> None:
    """Aggiunge al video lo stato corrente in modo leggibile."""

    text = (
        f"Distanza normalizzata: {distance:.3f}"
        if distance is not None
        else "Mano non rilevata"
    )
    cv2.rectangle(frame, (12, 12), (475, 58), (20, 20, 20), -1)
    cv2.putText(
        frame,
        text,
        (25, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
