"""
detect.py — Motion-gated vehicle detection with tracking.

Strategy:
  1. BackgroundSubtractorMOG2 finds moving regions each frame.
  2. Chronic-motion suppression eliminates shaking trees / flags (cells
     that fire in >50% of recent frames).
  3. ROI filter keeps only blobs on the road.
  4. Optional --verify: runs YOLO on each blob crop to confirm it's a vehicle.
  5. ByteTrack assigns persistent IDs so each car is followed entry → exit.
  6. Annotated video written via cv2.VideoWriter.

Usage:
    python detect.py --video raw_2026-04-05.mov
    python detect.py --video raw_2026-04-05.mov --verify             # + YOLO check
    python detect.py --video raw_2026-04-05.mov --verify --conf 0.08
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

# COCO vehicle classes
COCO_VEHICLE = {2, 3, 5, 7}

# Track colors — cycle through distinct hues for each ID
_PALETTE = [
    (50, 200, 50), (50, 150, 255), (255, 100, 50),
    (220, 50, 220), (50, 220, 220), (255, 200, 50),
]

def _track_color(tid: int) -> tuple:
    return _PALETTE[tid % len(_PALETTE)]


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
    check = [
        ((x1 + x2) // 2, (y1 + y2) // 2),
        ((x1 + x2) // 2, y2),
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
        self.counts *= (self.window - 1) / self.window
        for bx, by, bw, bh in blobs:
            gx, gy = self._cell(bx + bw / 2, by + bh / 2)
            self.counts[gy, gx] += 1.0
        survivors = []
        for blob in blobs:
            bx, by, bw, bh = blob
            gx, gy = self._cell(bx + bw / 2, by + bh / 2)
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
    ap.add_argument("--conf",    type=float, default=0.08)
    ap.add_argument("--pad",     type=float, default=0.4)
    ap.add_argument("--verify",  action="store_true",
                    help="Run YOLO on each motion crop to confirm it's a vehicle")
    args = ap.parse_args()

    # YOLO model — only loaded if verifying
    model = keep_cls = None
    if args.verify:
        from ultralytics import YOLO
        model = YOLO(args.model)
        names = model.names or {}
        keep_cls = None if len(names) == 1 else list(COCO_VEHICLE)
        print(f"Verify model: {args.model}  conf={args.conf}")

    roi_poly = load_roi(args.roi)
    print(f"ROI: {'loaded' if roi_poly is not None else 'none'}")
    roi_rect = cv2.boundingRect(roi_poly) if roi_poly is not None else None  # (x,y,w,h)

    cap   = cv2.VideoCapture(args.video)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}×{H} @ {fps:.1f} fps  ({total} frames)")

    fgbg        = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
    open_k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    chronic_map = ChronicMotionMap(W, H)
    tracker     = sv.ByteTrack(frame_rate=int(fps))
    trace_ann   = sv.TraceAnnotator(thickness=2, trace_length=int(fps * 3))
    img_area    = W * H

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    frame_idx = 0
    warmup    = 200

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

            raw_blobs = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < img_area * 0.0008 or area > img_area * 0.08:
                    continue
                raw_blobs.append(cv2.boundingRect(cnt))

            blobs = chronic_map.update_and_filter(raw_blobs)

            # ── Build detections ──────────────────────────────────────────────
            boxes, confs = [], []

            if frame_idx >= warmup:
                for bx, by, bw, bh in blobs:
                    if roi_poly is not None and not in_roi(roi_poly, bx, by, bx+bw, by+bh):
                        continue

                    conf_score = 1.0
                    if model is not None:
                        pad_x = int(bw * args.pad)
                        pad_y = int(bh * args.pad)
                        x1c = max(0, bx - pad_x);  y1c = max(0, by - pad_y)
                        x2c = min(W, bx+bw+pad_x); y2c = min(H, by+bh+pad_y)
                        crop = frame[y1c:y2c, x1c:x2c]
                        if crop.size == 0:
                            continue
                        res = model(crop, conf=args.conf, classes=keep_cls, verbose=False)[0]
                        if not res.boxes or len(res.boxes) == 0:
                            continue
                        conf_score = float(res.boxes.conf.max())

                    # Clip to ROI bounding rect
                    if roi_rect is not None:
                        rx, ry, rw, rh = roi_rect
                        fx1, fy1 = max(bx, rx),      max(by, ry)
                        fx2, fy2 = min(bx+bw, rx+rw), min(by+bh, ry+rh)
                    else:
                        fx1, fy1, fx2, fy2 = bx, by, bx+bw, by+bh

                    if fx2 > fx1 and fy2 > fy1:
                        boxes.append([fx1, fy1, fx2, fy2])
                        confs.append(conf_score)

            # ── ByteTrack ─────────────────────────────────────────────────────
            if boxes:
                dets = sv.Detections(
                    xyxy=np.array(boxes, dtype=float),
                    confidence=np.array(confs, dtype=float),
                    class_id=np.zeros(len(boxes), dtype=int),
                )
                dets = tracker.update_with_detections(dets)
            else:
                dets = sv.Detections.empty()

            # ── Draw ─────────────────────────────────────────────────────────
            ann = frame.copy()

            if roi_poly is not None:
                cv2.polylines(ann, [roi_poly], True, (0, 220, 120), 2)

            # Motion traces
            if len(dets) > 0:
                ann = trace_ann.annotate(ann, dets)

            # Boxes + IDs
            tids = dets.tracker_id if dets.tracker_id is not None else []
            for xyxy, tid in zip(dets.xyxy, tids):
                if tid is None:
                    continue
                x1, y1, x2, y2 = xyxy.astype(int)
                col = _track_color(int(tid))
                cv2.rectangle(ann, (x1, y1), (x2, y2), col, 2)
                lbl = f"#{tid}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(ann, (x1, y1 - th - 8), (x1 + tw + 4, y1), col, -1)
                cv2.putText(ann, lbl, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            status = f"frame {frame_idx}  |  {len(dets)} tracked  |  {len(blobs)} blobs"
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
    print(f"Tracks seen: {len(tracker.lost_tracks) + len(tracker.tracked_tracks)}")


if __name__ == "__main__":
    main()
