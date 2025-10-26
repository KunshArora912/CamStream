#!/usr/bin/env python3
# Jetson: ultra-light snapshot server
from flask import Flask, Response, jsonify
import cv2, time

DEV = "/dev/video0"     # USB cam
W, H = 640, 480         # drop to 320x240 if you want even lighter
JPEG_QUALITY = 60       # lower = smaller, faster

app = Flask(__name__)

cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS,          15)   # hint; snapshot is on-demand
if not cap.isOpened():
    raise SystemExit("Cannot open /dev/video0; check permissions (group 'video').")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "ts": time.time()}), 200

@app.route("/snapshot", methods=["GET"])
def snapshot():
    # Read a single frame on demand
    ok, frame = cap.read()
    if not ok or frame is None:
        time.sleep(0.01)
        return Response("no frame", status=503)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return Response("encode error", status=500)
    return Response(bytes(buf), mimetype="image/jpeg",
                    headers={"Cache-Control":"no-store, no-cache, must-revalidate, max-age=0"})

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5055, threaded=True)
    finally:
        cap.release()
