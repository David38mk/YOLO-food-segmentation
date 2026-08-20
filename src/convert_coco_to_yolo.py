import json
import os
import cv2
from tqdm import tqdm

# CHANGE THESE PATHS ONLY IF NEEDED
RAW_ROOT = "raw_data"
OUTPUT = "yolo_final"

SPLITS = {
    "train": "public_training_set_release_2.0",
    "val": "public_validation_set_2.0",
}

def yolo_bbox(box, img_w, img_h):
    x, y, w, h = box
    return (
        (x + w / 2) / img_w,
        (y + h / 2) / img_h,
        w / img_w,
        h / img_h
    )

def convert_split(split, folder):
    print(f"\nProcessing {split.upper()}...")

    img_src = os.path.join(RAW_ROOT, folder, "images")
    ann_path = os.path.join(RAW_ROOT, folder, "annotations.json")

    out_img = os.path.join(OUTPUT, "images", split)
    out_lbl = os.path.join(OUTPUT, "labels", split)

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    # Load annotations
    with open(ann_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    ann_map = {}

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        ann_map.setdefault(img_id, []).append(ann)

    corrupted = 0
    written = 0

    for img_id, meta in tqdm(images.items()):
        src = os.path.join(img_src, meta["file_name"])
        dst = os.path.join(out_img, meta["file_name"])
        label_file = os.path.join(out_lbl, meta["file_name"].replace(".jpg", ".txt"))

        if not os.path.exists(src):
            corrupted += 1
            continue

        image = cv2.imread(src)
        if image is None:
            corrupted += 1
            continue

        h, w = image.shape[:2]
        cv2.imwrite(dst, image)

        if img_id not in ann_map:
            open(label_file, "w").close()
            continue

        valid = []

        for a in ann_map[img_id]:
            cx, cy, bw, bh = yolo_bbox(a["bbox"], w, h)

            if bw <= 0 or bh <= 0: continue
            if cx < 0 or cy < 0 or cx > 1 or cy > 1: continue
            if bw > 1 or bh > 1: continue

            valid.append(f"0 {cx} {cy} {bw} {bh}")

        if valid:
            with open(label_file, "w") as f:
                f.write("\n".join(valid))
            written += 1
        else:
            open(label_file, "w").close()

    print(f"{split}: ✅ {written} labeled images")
    print(f"{split}: ⚠️ {corrupted} invalid images")

# RUN
for k, v in SPLITS.items():
    convert_split(k, v)

print("\n✅ CONVERSION COMPLETE → YOLO FOOD DATASET CREATED")
