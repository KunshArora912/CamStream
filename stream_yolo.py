#!/usr/bin/env python3
"""
yolo_from_snapshots.py
Pulls frames from a snapshot endpoint (e.g., Jetson Flask /snapshot),
runs YOLO, and posts the annotated JPEG to your backend /api/video/frame.

Usage:
  python yolo_from_snapshots.py \
    --snap-url http://<JETSON_IP>:5055/snapshot \
    --api-url  http://<BACKEND_HOST>:5000 \
    --room-id  roomA \
    --model    yolo11n.pt \
    --conf     0.25 \
    --fps      8
"""

import time
import io
import argparse
import requests
import numpy as np
import cv2

# Ultralytics (supports v8 and v11 models)
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-url", required=True, help="Snapshot endpoint, e.g. http://10.0.0.5:5055/snapshot")
    ap.add_argument("--api-url",  required=True, help="Backend base, e.g. http://localhost:5000")
    ap.add_argument("--room-id",  default="roomA")
    ap.add_argument("--model",    default="yolov8n.pt", help="Model path or name (yolov8n.pt, yolo11n.pt, etc.)")
    ap.add_argument("--conf",     type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou",      type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--imgsz",    type=int,   default=640, help="Inference size (longest edge)")
    ap.add_argument("--fps",      type=float, default=8.0, help="Target output fps")
    ap.add_argument("--timeout",  type=float, default=3.0, help="HTTP timeout seconds")
    ap.add_argument("--auth",     help="Bearer token if your backend is protected")
    ap.add_argument("--labels",   action="store_true", help="Draw labels on boxes (default True).")
    ap.add_argument("--no-labels",dest="labels", action="store_false", help="Do not draw labels")
    ap.set_defaults(labels=True)
    return ap.parse_args()

def fetch_snapshot(url: str, timeout: float) -> np.ndarray | None:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200 or "image" not in r.headers.get("Content-Type",""):
            return None
        data = np.frombuffer(r.content, dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def draw_with_ultralytics(model: YOLO, bgr: np.ndarray, imgsz: int, conf: float, iou: float, labels: bool) -> np.ndarray:
    # Ultralytics expects RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(
        source=rgb,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False
    )
    # results[0].plot() returns an annotated RGB image (np.ndarray)
    ann_rgb = results[0].plot(labels=labels)  # labels=True adds class+conf
    ann_bgr = cv2.cvtColor(ann_rgb, cv2.COLOR_RGB2BGR)
    return ann_bgr

def encode_jpeg(img: np.ndarray, quality: int = 70) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf) if ok else None

def post_latest(api_url: str, room_id: str, jpeg_bytes: bytes, token: str | None, timeout: float) -> bool:
    url = f"{api_url.rstrip('/')}/api/video/frame"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    files = {
        "frame": ("frame.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")
    }
    try:
        r = requests.post(f"{url}?roomId={room_id}", files=files, headers=headers, timeout=timeout)
        print(f"{url}?roomId={room_id}")
        return r.ok
    except Exception:
        return False

def main():
    args = parse_args()

    print(f"[YOLO] Loading model: {args.model}")
    model = YOLO(args.model)

    min_period = 1.0 / max(args.fps, 0.1)
##    print(f"[Worker] Starting loop: snap={args.snap-url} -> api={args.api_url} room={args.room_id} @ {args.fps} fps")
    print(f"[Worker] Starting loop: snap={args.snap_url} -> api={args.api_url} room={args.room_id} @ {args.fps} fps")

    backoff = 0.5
    while True:
        t0 = time.time()

        # 1) pull a frame
        frame = fetch_snapshot(args.snap_url, args.timeout)
        if frame is None:
            # could not get a frame; small backoff
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 2.0)
            continue
        backoff = 0.5

        # 2) YOLO
        try:
            annotated = draw_with_ultralytics(
                model, frame, args.imgsz, args.conf, args.iou, labels=args.labels
            )
        except Exception as e:
            print(f"[YOLO] inference error: {e}")
            time.sleep(0.25)
            continue

        # 3) JPEG encode
        jpeg = encode_jpeg(annotated, quality=70)
        if jpeg is None:
            time.sleep(0.1)
            continue

        # 4) POST to backend
        ok = post_latest(args.api_url, args.room_id, jpeg, args.auth, args.timeout)
        if not ok:
            # backend might be down; don’t spam
            time.sleep(0.5)

        # pacing to target fps
        dt = time.time() - t0
        if dt < min_period:
            time.sleep(min_period - dt)

if __name__ == "__main__":
    main()
