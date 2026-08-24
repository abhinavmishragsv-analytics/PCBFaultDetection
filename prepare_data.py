"""
Convert the DeepPCB dataset into YOLOv8-ready format.

DeepPCB layout:
    PCBData/trainval.txt   -> "group.../xxx.jpg group..._not/xxx.txt" (999 pairs)
    PCBData/test.txt       -> same format (499 pairs)

Each line's .jpg actually refers to the *_test.jpg (the defective image we
detect on; the *_temp.jpg is the clean template, not used for detection).

Each annotation .txt has one defect per line, space-separated:
    x1 y1 x2 y2 type
where type in {1..6}:
    1-open, 2-short, 3-mousebite, 4-spur, 5-copper(spurious copper), 6-pin-hole

We convert to YOLO format (class cx cy w h, all normalized 0-1, class 0-indexed)
and lay out:
    yolo_dataset/
        images/train/*.jpg   images/val/*.jpg
        labels/train/*.txt   labels/val/*.txt
"""

import shutil
from pathlib import Path
from PIL import Image

SRC_ROOT = Path("DeepPCB/PCBData")
OUT_ROOT = Path("yolo_dataset")

# DeepPCB type IDs are 1-indexed; YOLO wants 0-indexed classes in this order.
CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]


def resolve_image_path(rel_jpg_path: str) -> Path:
    """trainval.txt lists 'group.../name.jpg' but the actual file on disk
    is 'name_test.jpg' (the defective/tested image)."""
    p = Path(rel_jpg_path)
    fixed_name = p.stem + "_test" + p.suffix
    return SRC_ROOT / p.parent / fixed_name


def convert_annotation(txt_path: Path, img_w: int, img_h: int, out_txt_path: Path):
    lines_out = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x1, y1, x2, y2, cls_id = line.split()
            x1, y1, x2, y2, cls_id = float(x1), float(y1), float(x2), float(y2), int(cls_id)

            cls_idx = cls_id - 1  # DeepPCB is 1-indexed -> YOLO is 0-indexed
            if not (0 <= cls_idx < len(CLASS_NAMES)):
                continue  # skip malformed/background rows

            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            w = abs(x2 - x1) / img_w
            h = abs(y2 - y1) / img_h
            lines_out.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    out_txt_path.write_text("\n".join(lines_out))


def process_split(list_file: Path, split_name: str):
    img_out_dir = OUT_ROOT / "images" / split_name
    lbl_out_dir = OUT_ROOT / "labels" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_skipped, n_boxes = 0, 0, 0
    with open(list_file) as f:
        pairs = [line.strip().split() for line in f if line.strip()]

    for rel_img, rel_ann in pairs:
        img_path = resolve_image_path(rel_img)
        ann_path = SRC_ROOT / rel_ann

        if not img_path.exists() or not ann_path.exists():
            n_skipped += 1
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        stem = img_path.stem  # e.g. 00041001_test
        shutil.copy(img_path, img_out_dir / f"{stem}.jpg")
        out_lbl = lbl_out_dir / f"{stem}.txt"
        convert_annotation(ann_path, w, h, out_lbl)
        n_boxes += sum(1 for _ in open(out_lbl) if _.strip())
        n_ok += 1

    print(f"[{split_name}] converted={n_ok}  skipped={n_skipped}  total_boxes={n_boxes}")


def write_data_yaml():
    yaml_text = (
        f"path: {OUT_ROOT.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n"
        + "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
        + "\n"
    )
    Path("data.yaml").write_text(yaml_text)
    print("Wrote data.yaml")


if __name__ == "__main__":
    process_split(SRC_ROOT / "trainval.txt", "train")
    process_split(SRC_ROOT / "test.txt", "val")
    write_data_yaml()
