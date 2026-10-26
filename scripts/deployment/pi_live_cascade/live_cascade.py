"""Headless OV5647/Picamera2 cascade with an SSH-tunneled browser preview.

Example: python live_cascade.py --camera-only
         python live_cascade.py
         python live_cascade.py --source example.jpg --max-frames 1
"""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import platform
import threading
import time

import cv2


ROOT = Path(__file__).resolve().parent
PAGE = b'''<!doctype html><meta charset="utf-8"><title>Lily live preview</title>
<style>body{font:17px system-ui;background:#101820;color:#eee;max-width:1300px;margin:25px auto;padding:15px}
img{max-width:100%;background:#222}button{padding:12px;margin:10px 0}pre{white-space:pre-wrap}</style>
<h1>Lily live preview</h1><p id="status">Waiting for camera...</p><img src="/stream.mjpg">
<p>Blue: lily box. Yellow: candidate anther tip. Gray: flower skipped by the processing limit.</p>
<button onclick="save()">Save raw frame, preview and coordinates</button><span id="saved"></span>
<p>Coordinates refer to the displayed frame. Tracking: off.</p>
<script>async function update(){try{let s=await(await fetch('/status')).json();
let age=s.last_result_age_ms==null?'waiting':(s.last_result_age_ms/1000).toFixed(1)+'s ago';
document.getElementById('status').textContent=s.error||
((s.mode||'Starting')+' | '+(s.processed_flowers||0)+' flowers / '+(s.candidate_tips||0)+' candidate tips'+
' | processing '+(s.processing_fps||0).toFixed(1)+' FPS | frame age at completion '+
(s.frame_age_ms||0).toFixed(0)+' ms | last result '+age); }catch(e){
document.getElementById('status').textContent='Connection lost';}}
async function save(){let r=await fetch('/snapshot',{method:'POST'});let s=await r.json();
document.getElementById('saved').textContent=s.saved||s.error;}
setInterval(update,1000);update();</script>'''


class Hub:
    def __init__(self, folder):
        self.condition = threading.Condition()
        self.folder = folder
        self.jpeg = None
        self.raw = None
        self.status = {"mode": "Starting"}
        self.sequence = 0
        self.updated = None
        self.running = True

    def publish(self, raw, jpeg, status):
        with self.condition:
            self.raw, self.jpeg, self.status = raw, jpeg, status
            self.sequence += 1
            self.updated = time.monotonic()
            self.condition.notify_all()


def handler_for(hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, data, code=200):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(PAGE)))
                    self.end_headers()
                    self.wfile.write(PAGE)
                elif self.path == "/status":
                    with hub.condition:
                        status = dict(hub.status)
                        status["last_result_age_ms"] = ((time.monotonic() - hub.updated) * 1000
                                                        if hub.updated is not None else None)
                    self.send_json(status)
                elif self.path == "/stream.mjpg":
                    self.connection.settimeout(5)
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    seen = 0
                    while hub.running:
                        with hub.condition:
                            hub.condition.wait_for(lambda: hub.sequence != seen or not hub.running, timeout=2)
                            if not hub.running:
                                break
                            if hub.jpeg is None or seen == hub.sequence:
                                continue
                            seen, jpeg = hub.sequence, hub.jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def do_POST(self):
            if self.path != "/snapshot":
                self.send_error(404)
                return
            with hub.condition:
                raw, jpeg, status = hub.raw, hub.jpeg, dict(hub.status)
            if raw is None:
                self.send_json({"error": "No processed frame available yet"}, 503)
                return
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            prefix = hub.folder / f"snapshot_{stamp}"
            if not cv2.imwrite(str(prefix) + "_raw.jpg", raw):
                self.send_json({"error": "Could not save raw image"}, 500)
                return
            Path(str(prefix) + "_preview.jpg").write_bytes(jpeg)
            Path(str(prefix) + "_points.json").write_text(json.dumps(status, indent=2))
            self.send_json({"saved": str(prefix)})
    return Handler


class PiCamera:
    """Continuously capture into ONE replaceable slot, never a growing queue."""
    def __init__(self, index, width, height, fps):
        from picamera2 import Picamera2
        self.camera = Picamera2(index)
        # Picamera2 RGB888 arrays contain B,G,R bytes: correct for OpenCV/YOLO.
        config = self.camera.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps}, buffer_count=4, queue=False)
        self.camera.configure(config)
        self.width, self.height = self.camera.stream_configuration("main")["size"]
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.latest = None
        self.error = None
        self.camera.start()
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()

    def capture(self):
        number = 0
        try:
            while not self.stopping.is_set():
                request = self.camera.capture_request()
                try:
                    received = time.monotonic()
                    frame = request.make_array("main")[:self.height, :self.width, :3].copy()
                    metadata = request.get_metadata()
                finally:
                    request.release()
                with self.condition:
                    self.latest = (number, received, frame, metadata.get("SensorTimestamp"))
                    number += 1
                    self.condition.notify_all()
        except Exception as exc:
            if not self.stopping.is_set():
                with self.condition:
                    self.error = str(exc)
                    self.condition.notify_all()

    def get(self, previous):
        with self.condition:
            ready = self.condition.wait_for(lambda:
                self.error is not None or (self.latest is not None and self.latest[0] != previous), timeout=5)
            if self.error:
                raise RuntimeError("Camera capture failed: " + self.error)
            if not ready:
                raise RuntimeError("No new camera frame for five seconds")
            return self.latest

    def close(self):
        self.stopping.set()
        self.camera.stop()
        self.thread.join(timeout=2)
        self.camera.close()


class FileSource:
    """Offline image/video replay for checking the exact same inference code."""
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.still = self.path.suffix.lower() in (".jpg", ".jpeg", ".png")
        self.capture = None if self.still else cv2.VideoCapture(str(self.path))
        if self.capture is not None and not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")
        self.number = 0

    def get(self, previous):
        if self.still:
            if self.number:
                raise EOFError
            frame = cv2.imread(str(self.path))
            if frame is None:
                raise ValueError("Could not decode image")
        else:
            ok, frame = self.capture.read()
            if not ok:
                raise EOFError
        value = (self.number, time.monotonic(), frame, None)
        self.number += 1
        return value

    def close(self):
        if self.capture:
            self.capture.release()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="picamera2", help="picamera2, or an image/video path")
    parser.add_argument("--camera-only", action="store_true", help="Test capture/preview without loading models")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1296)
    parser.add_argument("--height", type=int, default=972)
    parser.add_argument("--camera-fps", type=float, default=15)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lily-model", type=Path, default=ROOT / "models/lily.pt")
    parser.add_argument("--anther-model", type=Path, default=ROOT / "models/anther.pt")
    parser.add_argument("--lily-confidence", type=float, default=.60)
    parser.add_argument("--peak-threshold", type=float, default=.80)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--crop-margin", type=float, default=.10)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--max-flowers", type=int, default=3, help="Per-frame heatmap limit; 0 processes all")
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs until Ctrl+C")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main(args=None):
    args = args or arguments()
    if not (0 < args.lily_confidence <= 1 and 0 < args.peak_threshold <= 1):
        raise ValueError("Confidence/peak thresholds must be in (0, 1]")
    if min(args.width, args.height, args.threads) < 1 or args.camera_fps <= 0:
        raise ValueError("Dimensions, threads and camera FPS must be positive")
    if args.max_flowers < 0 or args.max_frames < 0 or args.crop_margin < 0:
        raise ValueError("Limits and margin must be nonnegative")
    cv2.setNumThreads(1)
    cascade = None
    if not args.camera_only:
        from cascade_runtime import Cascade
        print("Loading cascade models...", flush=True)
        cascade = Cascade(args.lily_model, args.anther_model, args.lily_confidence,
                          args.peak_threshold, args.image_size, args.crop_margin,
                          args.max_flowers, args.threads)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = args.output or ROOT / "sessions" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(python=platform.python_version(), architecture=platform.machine(),
                  heatmap_epoch=cascade.checkpoint_epoch if cascade else None)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    hub = Hub(folder)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(hub))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Preview: http://localhost:{args.port} (use an SSH tunnel from your Mac)", flush=True)
    print("Session:", folder, flush=True)
    source = None
    last = -1
    count = 0
    previous_finished = None
    last_preview_save = -float("inf")
    try:
        source = PiCamera(args.camera_index, args.width, args.height, args.camera_fps) if args.source == "picamera2" else FileSource(args.source)
        with (folder / "observations.jsonl").open("w") as log:
            while not args.max_frames or count < args.max_frames:
                try:
                    sequence, received, raw, sensor_timestamp = source.get(last)
                except EOFError:
                    break
                last = sequence
                started = time.monotonic()
                if cascade:
                    annotated, status = cascade.process(raw)
                else:
                    annotated = raw.copy()
                    status = {"frame_wh": [raw.shape[1], raw.shape[0]], "flowers": [],
                              "processed_flowers": 0, "candidate_tips": 0}
                inference_finished = time.monotonic()
                process_ms = (inference_finished - started) * 1000
                status.update(mode="Cascade" if cascade else "Camera preview",
                              frame_sequence=sequence,
                              sensor_timestamp_ns=(int(sensor_timestamp) if sensor_timestamp is not None else None),
                              frame_age_ms=(inference_finished - received) * 1000,
                              inference_fps=(1000 / process_ms if process_ms and cascade else 0),
                              processing_fps=(1 / (inference_finished - previous_finished)
                                              if previous_finished is not None else 0),
                              timestamp_utc=datetime.now(timezone.utc).isoformat())
                previous_finished = inference_finished
                text = (f"{status['mode']} | flowers {status['processed_flowers']} | "
                        f"tips {status['candidate_tips']} | {status['processing_fps']:.1f} processing FPS")
                cv2.putText(annotated, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            .6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            .6, (255, 255, 255), 1, cv2.LINE_AA)
                ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise RuntimeError("Preview JPEG encoding failed")
                hub.publish(raw, buffer.tobytes(), status)
                log.write(json.dumps(status) + "\n")
                log.flush()
                count += 1
                if count == 1 or count % 20 == 0:
                    print(text + f" | frame age {status['frame_age_ms']:.0f} ms", flush=True)
                # Always retain the most recent result for offline testing and inspection.
                if inference_finished - last_preview_save >= 2:
                    (folder / "latest_preview.jpg").write_bytes(buffer.tobytes())
                    last_preview_save = inference_finished
    except KeyboardInterrupt:
        print("Stopping...", flush=True)
    finally:
        with hub.condition:
            hub.running = False
            hub.condition.notify_all()
        try:
            if source:
                source.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
    print("Saved session:", folder, flush=True)


if __name__ == "__main__":
    main()
