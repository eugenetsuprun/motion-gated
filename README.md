# motion-gated

Vehicle detection from a fixed overhead security camera using background subtraction — no model training required.

## How it works

1. **Background subtraction** (MOG2) finds moving pixels each frame
2. **Chronic-motion suppression** eliminates shaking trees / flags (cells firing in >50% of recent frames)
3. **ROI filter** restricts detections to the road polygon
4. **Blob bounding boxes** are the detections — no neural network needed

## Usage

```bash
python detect.py --video raw.mov
python detect.py --video raw.mov --roi roi.json --output out.mp4
```

## TODO

- [ ] **Auto-detect roadway**: use a segmentation model (e.g. SegFormer or SAM) to automatically identify the road polygon from the first frame, eliminating the need to manually draw the ROI
- [ ] Merge fragmented blobs from the same car (headlight false positives)
- [ ] Add ByteTrack for persistent vehicle IDs across frames
- [ ] Speed estimation using road centerline calibration
