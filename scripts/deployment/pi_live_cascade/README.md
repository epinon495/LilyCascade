# Lily cascade live test on Raspberry Pi

Configured for the detected OV5647 ribbon camera on 64-bit Raspberry Pi OS Lite
Trixie. The camera is already detected by `rpicam-hello --list-cameras`; no
additional camera overlay or driver change is prescribed here.

This package contains the trained YOLO26n detector and the epoch-108 augmented
anther heatmap checkpoint. The anther checkpoint contains inference weights and
configuration only; optimizer buffers were removed without changing weights.
`model_manifest.json` records their origins and SHA-256 checksums.

## 1. Transfer the package from the Mac

Replace `YOUR_USER@YOUR_PI` with the username and address you already use for SSH.
In a Mac Terminal:

```sh
scp "/Users/evpinon/Documents/Codex/2026-10-10/for/outputs/pi_live_cascade.zip" YOUR_USER@YOUR_PI:~/
```

## 2. Open an SSH connection with a preview tunnel

In a Mac Terminal:

```sh
ssh -L 127.0.0.1:8000:127.0.0.1:8000 YOUR_USER@YOUR_PI
```

Leave this connection open. Run the following setup commands **on the Pi**, in
that SSH session. The browser will run on the Mac. No desktop or X forwarding is
required on the Pi. If Mac port 8000 is busy, use local port 8001 on the left side
of the tunnel and open `http://localhost:8001` instead.

## 3. Install on the Pi

```sh
python3 -m zipfile -e pi_live_cascade.zip .
cd pi_live_cascade
sudo apt update
sudo apt install --no-install-recommends python3-venv python3-picamera2 libgl1 libglib2.0-0t64
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the Pi's system Python. `--system-site-packages` makes the matching apt-installed
Picamera2/libcamera bindings visible inside the environment. Picamera2 is not
installed from pip. The requirements are for Trixie's Python 3.13 and NumPy 2
camera stack; they keep NumPy below 2.3 and pin OpenCV and Ultralytics. PyTorch and
torchvision are resolved together for the Pi's architecture. The environment is
created on the Pi, so the libraries match Linux ARM64.

## 4. Verify the camera stream first

```sh
python live_cascade.py --camera-only
```

On the Mac, open **http://localhost:8000**. Check colors, orientation, focus and
lighting. Stop with Ctrl+C in the Pi terminal when the preview works. Close any
other application using the camera before starting this script.

## 5. Run the cascade

```sh
python live_cascade.py
```

Open or refresh **http://localhost:8000** on the Mac.

- Blue boxes: accepted lily detections.
- Yellow numbered circles: candidate anther-tip peaks.
- Gray boxes: accepted lilies skipped by the per-frame heatmap processing limit.
- The page reports processed flowers, candidate tips, measured output cadence,
  frame age at inference completion, and time since the last result.

Defaults: 1296 x 972 camera frame, 15 requested camera FPS, YOLO input 416,
lily confidence 0.60, 10% crop margin, 256 x 256 heatmap, peak threshold 0.80,
four-pixel minimum peak spacing, up to three flowers processed per frame.
The heatmap limit selects the highest lily scores, **not the nearest flowers**.
Each heatmap point maps back to coordinates in that camera frame.

The camera capture loop continually replaces one latest-frame slot while
inference runs. Older frames are discarded instead of accumulating in a queue.
The browser displays the exact frame used for each result, with its own boxes
and points. It does not draw old predictions onto a newer frame.

## Controls and settings

For temperature, fan telemetry, throttling flags and stage timings, open another
SSH session and run `python3 monitor_pi.py` from this folder. The monitor discovers
fan interfaces automatically: RPM is measured feedback, while PWM percentage and
cooling state describe requested drive. A missing sensor reports unavailable,
not that the fan is physically off. No fan settings are changed. `--once` produces
a single JSON report. RPM feedback can lag control changes briefly.

Press **Save raw frame, preview and coordinates** in the browser to save a matched
set of files into the session folder on the Pi. The raw image has no boxes drawn
on it, so it is suitable for later labeling. The preview JPEG and JSON coordinates
refer to that same source frame.

Every invocation creates a new `sessions/TIMESTAMP` folder containing settings,
`observations.jsonl`, and a periodically updated `latest_preview.jpg`. Snapshot
files are created when you press the button. The script does not record full video.

Example configuration changes:

```sh
python live_cascade.py --peak-threshold 0.7
python live_cascade.py --max-flowers 1
python live_cascade.py --max-flowers 0
python live_cascade.py --width 640 --height 480
python live_cascade.py --threads 2
python live_cascade.py --help
```

Reducing camera dimensions can remove detail from the small tips, even though
the heatmap input remains 256 x 256. Start with the default and measure speed.
For the initial test, avoid changing several settings together.

## Offline check

To confirm model loading and inference without accessing the camera:

```sh
python live_cascade.py --source example.jpg --max-frames 1
```

This exits after one image. Inspect the saved `latest_preview.jpg` and observation
log; the browser server stops when the run ends. Video paths are supported too;
replay processes sequential frames as fast as it can, rather than honoring video
timestamps.

## Timing and current limits

The displayed processing FPS is the cadence of processed results, including the
previous cycle's encoding, capture waits and file work. Its first value is zero
because two results are required to measure a cadence. `inference_fps` in JSON
is a separate compute-only estimate.

Frame age starts when the capture thread receives the request. It includes
waiting for inference and inference itself, but excludes sensor exposure/readout,
JPEG encoding and browser/network delay. Sensor timestamps are also retained.
File replay age is an offline processing measurement, not camera latency.

This version provides anther candidates and lily boxes. It has no stigma channel,
cross-frame tracking, cross-crop point merging, depth estimation, target
confirmation or autopilot connection. Flower IDs are local to each frame.
Heatmap scores are not calibrated probabilities.

The Python inference, coordinate mapping, offline path and web handlers are
checked on the Mac. Actual OV5647 capture and Pi speed must be checked on the Pi.
Once the first live test works, NCNN is an option for optimizing YOLO on ARM;
compare exported predictions against this reference before switching formats.

## Sources

- Raspberry Pi Picamera2 manual: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
- Raspberry Pi camera software: https://www.raspberrypi.com/documentation/computers/camera_software.html
- Ultralytics Raspberry Pi guide: https://docs.ultralytics.com/guides/raspberry-pi/

The Picamera2 manual specifies that `RGB888` arrays contain B,G,R bytes. That is
the format used here for OpenCV and YOLO; the heatmap path explicitly converts
to RGB before applying the same image preparation used during training.
