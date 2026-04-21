"""
detect.py — Motion-gated vehicle detection.

Strategy:
  1. BackgroundSubtractorMOG2 finds moving regions each frame.
  2. Chronic-motion suppression eliminates shaking trees / flags (cells
     that fire in >50% of recent frames).
  3. YOLO runs only on padded crops around surviving blobs — not the full frame.
  4. Detections are mapped back to full-frame coords and ROI-filtered.
  5. Annotated video written via ffmpeg.

Usage:
    python detect.py --video raw_2026-04-05.mov
    python detect.py --video raw_2026-04-05.mov --model yolov8m.pt --conf 0.1
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# COCO vehicle classes
COCO_VEHICLE = {2, 3, 5, 7}

# Annotator colors (BGR)
_BOX   = (50, 200, 50)
_LABEL = (0, 0, 0)


# ── ROI ───────────────────────────────────────────────────────────────────────

def load_roi(path: str) -> np.ndarray | None:
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    pts = data.get("points") or data.get("road_polygon_px")
    if not pts:
        return None
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def in_roi(poly: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    # Accept if any of center, bottom-center, or corners is inside the polygon.
    # Strict centroid-only check misses boxes that straddle the ROI boundary.
    check = [
        ((x1 + x2) // 2, (y1 + y2) // 2),  # center
        ((x1 + x2) // 2, y2),               # bottom-center
        (x1, y1), (x2, y1),
        (x1, y2), (x2, y2),
    ]
    return any(
        cv2.pointPolygonTest(poly, (float(px), float(py)), False) >= 0
        for px, py in check
    )


# ── Chronic-motion suppression ────────────────────────────────────────────────

class ChronicMotionMap:
    """Suppress cells where motion blobs appear in >rate of recent frames."""

    def __init__(self, W: int, H: int, grid: int = 64, window: int = 150, rate: float = 0.5):
        self.cell_w = W / grid
        self.cell_h = H / grid
        self.grid   = grid
        self.window = window
        self.rate   = rate
        self.counts = np.zeros((grid, grid), dtype=np.float32)

    def _cell(self, cx: float, cy: float) -> tuple[int, int]:
        gx = min(int(cx / self.cell_w), self.grid - 1)
        gy = min(int(cy / self.cell_h), self.grid - 1)
        return gx, gy

    def update_and_filter(self, blobs: list[tuple]) -> list[tuple]:
        """Update counts with current blobs, return those that aren't chronic."""
        self.counts *= (self.window - 1) / self.window
        for bx, by, bw, bh in blobs:
            gx, gy = self._cell(bx + bw / 2, by + bh / 2)
            self.counts[gy, gx] += 1.0
        survivors = []
        for blob in blobs:
            bx, by, bw, bh = blob
            gx, gy = self._cell(bx + bw / 2, by + bh / 2)
            # counts converge to ~window at steady state, so normalize before comparing
            if self.counts[gy, gx] / self.window < self.rate:
                survivors.append(blob)
        return survivors


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",   default="raw_2026-04-05.mov")
    ap.add_argument("--output",  default="output.mp4")
    ap.add_argument("--model",   default="yolo26n.pt")
    ap.add_argument("--roi",     default="roi.json")
    ap.add_argument("--conf",    type=float, default=0.10)
    ap.add_argument("--pad",     type=float, default=0.3,
                    help="Fractional padding around each motion blob before YOLO crop")
    args = ap.parse_args()

    roi_poly = load_roi(args.roi)
    print(f"ROI: {'loaded' if roi_poly is not None else 'none'}")

    cap  = cv2.VideoCapture(args.video)
    fps  = cap.get(cv2.CAP_PROP_FPS)
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}×{H} @ {fps:.1f} fps  ({total} frames)")

    fgbg         = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
    open_k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate_k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    chronic_map  = ChronicMotionMap(W, H)
    img_area     = W * H

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    frame_idx = 0
    warmup    = 200  # frames before motion recovery kicks in

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ── BGS ──────────────────────────────────────────────────────────
            fg = fgbg.apply(frame)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  open_k)
            fg = cv2.dilate(fg, dilate_k, iterations=2)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Collect car-sized blobs
            raw_blobs = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < img_area * 0.0008 or area > img_area * 0.08:
                    continue
                raw_blobs.append(cv2.boundingRect(cnt))  # (x, y, w, h)

            # Chronic-motion suppression (shaking trees)
            blobs = chronic_map.update_and_filter(raw_blobs)

            # ── Motion blobs are the detections — no YOLO needed ─────────────
            ann = frame.copy()
            detections = []

            if frame_idx >= warmup:
                for bx, by, bw, bh in blobs:
                    if roi_poly is not None and not in_roi(roi_poly, bx, by, bx+bw, by+bh):
                        continue
                    # Clip box to ROI bounding rect so it never visually escapes the polygon
                    rx, ry, rw, rh = cv2.boundingRect(roi_poly)
                    fx1 = max(bx, rx)
                    fy1 = max(by, ry)
                    fx2 = min(bx + bw, rx + rw)
                    fy2 = min(by + bh, ry + rh)
                    if fx2 > fx1 and fy2 > fy1:
                        detections.append((fx1, fy1, fx2, fy2, 1.0))

            # ── Draw ─────────────────────────────────────────────────────────
            if roi_poly is not None:
                cv2.polylines(ann, [roi_poly], True, (0, 220, 120), 2)

            for fx1, fy1, fx2, fy2, conf in detections:
                cv2.rectangle(ann, (fx1, fy1), (fx2, fy2), _BOX, 2)
                lbl = f"{conf:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(ann, (fx1, fy1 - th - 6), (fx1 + tw + 4, fy1), _BOX, -1)
                cv2.putText(ann, lbl, (fx1 + 2, fy1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            status = f"frame {frame_idx}  |  {len(detections)} vehicle(s)  |  {len(blobs)} motion blobs"
            cv2.putText(ann, status, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(ann, status, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            writer.write(ann)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  {frame_idx}/{total}  ({frame_idx/total*100:.0f}%)")

    finally:
        cap.release()
        writer.release()

    print(f"\nOutput: {args.output}")


def _nms(dets: list[tuple], iou_thresh: float = 0.4) -> list[tuple]:
    """Simple greedy NMS to remove duplicate boxes from overlapping crops."""
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    kept = []
    for d in dets:
        x1, y1, x2, y2, _ = d
        duplicate = False
        for k in kept:
            kx1, ky1, kx2, ky2, _ = k
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter
            if union > 0 and inter / union > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    return kept


if __name__ == "__main__":
    main()
