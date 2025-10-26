#!/usr/bin/env python3
import time, json, requests, cv2, numpy as np
from ultralytics import YOLO

JETSON_SNAPSHOT = "http://10.11.81.246:5055/snapshot"      # <-- set IP
POST_STATUS     = None  # e.g., "http://<JETSON_IP>:5055/status_update"
FPS_LIMIT       = 8     # pull rate cap (Hz)

model = YOLO("yolov8n.pt")   # tiny model
CONF  = 0.4
HOLD  = 30                    # frames to keep occupied after last detection

occupied = False
hold = 0
last_pull = 0

def pull_jpeg(url, to=5):
    r = requests.get(url, timeout=to)
    r.raise_for_status()
    return r.content

while True:
    # simple pull rate limit
    now = time.time()
    dt = 1.0 / max(1, FPS_LIMIT)
    if now - last_pull < dt:
        time.sleep(dt - (now - last_pull))
    last_pull = time.time()

    try:
        jpg = pull_jpeg(JETSON_SNAPSHOT, to=3)
    except Exception as e:
        print("Snapshot error:", e)
        continue

    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue

    # YOLO inference
    res = model.predict(img, imgsz=480, conf=CONF, verbose=False)[0]

    # count person class (0)
    persons = sum(int(b.cls) == 0 for b in res.boxes)

    if persons > 0:
        occupied = True
        hold = HOLD
    else:
        hold = max(0, hold - 1)
        if hold == 0:
            occupied = False

    # draw for local debug
    for b in res.boxes:
        if int(b.cls) != 0:  # keep only people
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{float(b.conf):.2f}", (x1, max(0, y1-7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.putText(img, f"OCCUPIED: {occupied}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if occupied else (0,200,0), 2)
    cv2.imshow("YOLO on Laptop (snapshots)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # optional: post status back to Jetson for your website
    if POST_STATUS:
        try:
            requests.post(POST_STATUS,
                          json={"occupied": occupied, "ts": time.time(), "source": "laptop-yolo"},
                          timeout=0.5)
        except Exception:
            pass

cv2.destroyAllWindows()
