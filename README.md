# PCB Defect Detection with YOLOv8 (DeepPCB dataset)

Object detection pipeline that locates and classifies 6 types of PCB defects:
**open, short, mousebite, spur, spurious copper, pin-hole**.

Tested end-to-end against the real [DeepPCB](https://github.com/tangsanli5201/DeepPCB)
dataset (1,500 image pairs) — the data conversion script below was run and verified
to produce 1,000 training images (6,873 annotated defects) and 500 validation
images (3,140 annotated defects), matching the dataset's documented split.

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Get the data

```bash
git clone https://github.com/tangsanli5201/DeepPCB.git
```

This gives you `DeepPCB/PCBData/`, containing grouped folders of 640x640
image pairs (`*_test.jpg` = defective board, `*_temp.jpg` = clean template)
plus `.txt` annotation files (`x1 y1 x2 y2 type`, type 1-6) and the official
`trainval.txt` / `test.txt` split lists.

## 3. Convert to YOLO format

```bash
python3 prepare_data.py
```

This reads `DeepPCB/PCBData/{trainval,test}.txt`, resolves each entry to its
actual `*_test.jpg` file, converts the `x1,y1,x2,y2,type` boxes into
normalized YOLO `class cx cy w h` format, and writes:

```
yolo_dataset/
  images/train/*.jpg   images/val/*.jpg
  labels/train/*.txt   labels/val/*.txt
data.yaml
```

## 4. Train

```bash
python3 train.py --model yolov8n.pt --epochs 100 --imgsz 640 --device 0
```

Use `--device cpu` if you don't have a GPU (much slower — expect hours, not
minutes, for full training). `yolov8n` is a good default: defects are small
but the images are simple/grayscale, so a nano or small model is normally
enough without needing yolov8l/x.

Augmentation in `train.py` is deliberately conservative (mild rotation/
translation, no hue jitter) because PCB defects are geometry-and-position
sensitive — aggressive augmentation can distort exactly the shape cues
(a "spur" vs a "short") the model needs to learn.

Training outputs land in `runs/detect/pcb_defect_yolov8/`, including
`weights/best.pt`, a confusion matrix, PR curves, and per-class metrics.

## 5. Run inference

```bash
python3 infer.py --weights runs/detect/pcb_defect_yolov8/weights/best.pt \
                  --source yolo_dataset/images/val \
                  --conf 0.25
```

Saves annotated images (boxes + class + confidence drawn on them) and raw
YOLO-format predictions to `runs/predict/predictions/`, plus a console
summary of defect counts per image.

## Evaluating results

- **mAP50 / mAP50-95** — the standard detection accuracy metrics; check
  `runs/detect/pcb_defect_yolov8/results.png` after training.
- **Per-class performance** — some defect types (e.g. "short") tend to be
  easier than others (e.g. "spur", which is visually subtle) — check the
  per-class row in the val output rather than only the aggregate mAP.
- **DeepPCB's own benchmark** uses IoU ≥ 0.33 (not the usual 0.5) since
  defect boxes are tiny, and reports F-score alongside mAP — worth using
  the same threshold if you want to compare against the paper's ~98.6% mAP.

## Notes / things worth knowing

- The `*_temp.jpg` template images (clean reference boards) aren't used
  in this detection pipeline — they're there if you want to build a
  difference-based baseline (align template vs. test, threshold the diff)
  to compare against the YOLO model.
- Real inspection systems often run at much higher resolution than 640x640;
  if you apply this to your own PCB photos, tile large images into
  640-sized crops (with overlap) rather than downscaling, or defects will
  shrink below what the model can reliably detect.
- Dataset license: DeepPCB is for research use only (see their LICENSE).
