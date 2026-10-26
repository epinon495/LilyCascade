"""Portable CPU cascade. Camera and HTTP code live in live_cascade.py."""
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from heatmap_model import TinyUNet, find_peaks


class Cascade:
    def __init__(self, lily_model, anther_model, lily_confidence=0.6,
                 peak_threshold=0.8, image_size=416, crop_margin=0.1,
                 max_flowers=3, threads=3):
        for path in (Path(lily_model), Path(anther_model)):
            if not path.exists():
                raise FileNotFoundError(path)
        torch.set_num_threads(threads)
        cv2.setNumThreads(1)
        self.detector = YOLO(str(lily_model), task="detect")
        checkpoint = torch.load(anther_model, map_location="cpu", weights_only=True)
        self.heatmap = TinyUNet(checkpoint["config"]["MODEL_WIDTH"])
        self.heatmap.load_state_dict(checkpoint["model_state"])
        self.heatmap.eval()
        self.checkpoint_epoch = checkpoint.get("epoch")
        self.lily_confidence = lily_confidence
        self.peak_threshold = peak_threshold
        self.image_size = image_size
        self.crop_margin = crop_margin
        self.max_flowers = max_flowers

    def prepare_crop(self, image, box):
        x0, y0, x1, y1 = box
        width, height = x1 - x0, y1 - y0
        margin = self.crop_margin
        left, top = max(0, math.floor(x0 - margin * width)), max(0, math.floor(y0 - margin * height))
        right = min(image.width, math.ceil(x1 + margin * width))
        bottom = min(image.height, math.ceil(y1 + margin * height))
        if right <= left or bottom <= top:
            raise ValueError("Empty detected crop.")
        crop = image.crop((left, top, right, bottom))
        scale = 256 / max(crop.size)
        rw, rh = max(1, round(crop.width * scale)), max(1, round(crop.height * scale))
        px, py = (256 - rw) // 2, (256 - rh) // 2
        canvas = Image.new("RGB", (256, 256), (114, 114, 114))
        canvas.paste(crop.resize((rw, rh), Image.Resampling.BILINEAR), (px, py))
        return canvas, {"crop_xyxy": [left, top, right, bottom],
                        "resized_wh": [rw, rh], "padding_xy": [px, py],
                        "scale_xy": [rw / crop.width, rh / crop.height]}

    @torch.inference_mode()
    def process(self, bgr):
        """Return an annotated copy and pixel-coordinate observations.

        All boxes and points refer to THIS input frame. No tracking, target
        confirmation, distance estimation, or flight commands are performed.
        """
        start = time.monotonic()
        result = self.detector.predict(source=bgr, imgsz=self.image_size,
                                       conf=self.lily_confidence, device="cpu",
                                       rect=False, max_det=50, verbose=False)[0]
        detection_ms = (time.monotonic() - start) * 1000
        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        order = sorted(range(len(boxes)), key=lambda i: -scores[i])
        selected = order[:self.max_flowers] if self.max_flowers else order
        selected_set = set(selected)
        annotated = bgr.copy()
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        flowers = []
        for index in order:
            box = boxes[index]
            x0, y0, x1, y1 = map(round, box)
            color = (255, 160, 0) if index in selected_set else (140, 140, 140)
            cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
            label = f"F{index} {scores[index]:.2f}" + (" skipped" if index not in selected_set else "")
            cv2.putText(annotated, label, (x0, max(18, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if index not in selected_set:
                continue
            prepared, geometry = self.prepare_crop(rgb, box)
            array = np.asarray(prepared, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())[None]
            heatmap = self.heatmap(tensor).sigmoid()[0, 0].numpy()
            px, py = geometry["padding_xy"]
            rw, rh = geometry["resized_wh"]
            valid_heatmap = np.zeros_like(heatmap)
            valid_heatmap[py:py + rh, px:px + rw] = heatmap[py:py + rh, px:px + rw]
            sx, sy = geometry["scale_xy"]
            left, top, _, _ = geometry["crop_xyxy"]
            points = []
            for number, (x, y, score) in enumerate(find_peaks(valid_heatmap, self.peak_threshold, 4), 1):
                ox, oy = (x - px) / sx + left, (y - py) / sy + top
                points.append({"point_number": number, "xy": [ox, oy], "score": score})
                center = (round(ox), round(oy))
                cv2.circle(annotated, center, 5, (0, 255, 255), 2)
                cv2.putText(annotated, f"{index}:{number}", (center[0] + 5, center[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1, cv2.LINE_AA)
            flowers.append({"flower_number": index, "lily_score": scores[index],
                            "box_xyxy": box, "geometry": geometry, "points": points})
        total_ms = (time.monotonic() - start) * 1000
        return annotated, {"frame_wh": [bgr.shape[1], bgr.shape[0]],
                           "detected_flowers": len(boxes), "processed_flowers": len(flowers),
                           "skipped_flowers": len(boxes) - len(flowers),
                           "candidate_tips": sum(len(f["points"]) for f in flowers),
                           "detection_ms": detection_ms, "heatmap_and_drawing_ms": total_ms - detection_ms,
                           "cascade_ms": total_ms, "flowers": flowers,
                           "tracking_enabled": False}
