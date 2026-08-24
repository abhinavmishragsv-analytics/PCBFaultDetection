import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to trained .pt weights")
    parser.add_argument("--source", required=True, help="image file or directory")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--out", default="predictions", help="output directory name")
    args = parser.parse_args()

    model = YOLO(args.weights)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,             # saves annotated images with boxes drawn
        save_txt=True,         # also saves raw YOLO-format predictions
        save_conf=True,
        project="runs/predict",
        name=args.out,
    )

    # Print a quick per-image summary of what was found
    for r in results:
        img_name = Path(r.path).name
        counts = {}
        for box in r.boxes:
            cls_name = r.names[int(box.cls)]
            counts[cls_name] = counts.get(cls_name, 0) + 1
        summary = ", ".join(f"{k}:{v}" for k, v in counts.items()) or "no defects found"
        print(f"{img_name}: {summary}")

    print(f"\nAnnotated images saved to runs/predict/{args.out}/")


if __name__ == "__main__":
    main()
