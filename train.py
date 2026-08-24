"""
Train YOLOv8 on the converted DeepPCB defect dataset.

Usage:
    python3 train.py                      # sensible defaults
    python3 train.py --epochs 150 --model yolov8s.pt --imgsz 640

Notes on model size choice for this dataset:
    - Defects are small (often <5% of the 640x640 image) and grayscale,
      so yolov8n/yolov8s are usually enough -- no need for yolov8l/x here.
    - PCB images are effectively single-channel; YOLO will still run them
      as 3-channel (replicated), which is fine and lets you use standard
      pretrained weights.
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="yolov8n.pt (fastest) / yolov8s.pt (more accurate)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' for GPU 0, 'cpu' for CPU")
    parser.add_argument("--name", default="pcb_defect_yolov8")
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=20,          # early stopping
        # Defects are position/orientation-sensitive on a fixed board layout,
        # so keep geometric augmentation mild -- aggressive flips/rotations
        # can distort what "open"/"short"/"spur" look like relative to traces.
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        hsv_h=0.0,             # grayscale-ish images -> hue jitter is meaningless
        hsv_s=0.0,
        hsv_v=0.3,
    )

    # Run final validation and print per-class metrics (precision/recall/mAP)
    metrics = model.val()
    print(metrics)


if __name__ == "__main__":
    main()
