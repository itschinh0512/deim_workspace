"""
Convert YOLO-format labels in dataset_normal → COCO JSON annotations.

Usage:
    python tools/yolo2coco_normal.py

Outputs:
    /home/locth/omni2rect_DEIM/dataset_normal/annotations/instances_train.json
    /home/locth/omni2rect_DEIM/dataset_normal/annotations/instances_val.json
"""

import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset_normal"
CATEGORIES = [{"id": 0, "name": "license_plate", "supercategory": "object"}]
SPLITS = ["train", "val"]


def yolo_bbox_to_coco(cx_norm, cy_norm, w_norm, h_norm, img_w, img_h):
    """Convert normalised YOLO cx/cy/w/h → COCO [x_min, y_min, w, h] (absolute pixels)."""
    w = w_norm * img_w
    h = h_norm * img_h
    x_min = (cx_norm * img_w) - w / 2
    y_min = (cy_norm * img_h) - h / 2
    return [round(x_min, 4), round(y_min, 4), round(w, 4), round(h, 4)]


def build_coco_json(split: str) -> dict:
    img_dir = DATASET_ROOT / "images" / split
    lbl_dir = DATASET_ROOT / "labels" / split

    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"[{split}] found {len(img_files)} images")

    for img_path in tqdm(img_files, desc=split):
        # --- image record ---
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        # --- annotation records ---
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    # dataset_normal is a single-class dataset. DEIM expects
                    # labels in [0, num_classes), so the COCO category_id must
                    # be 0 when num_classes is 1.
                    cls_id = 0
                    cx, cy, bw, bh = map(float, parts[1:5])
                    bbox = yolo_bbox_to_coco(cx, cy, bw, bh, img_w, img_h)
                    area = round(bbox[2] * bbox[3], 4)

                    # COCO category id = cls_id (0-indexed, matching DEIMv2 convention)
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [],
                    })
                    ann_id += 1

        img_id += 1

    return {
        "info": {"description": "dataset_normal license plate detection", "version": "1.0"},
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


def main():
    ann_dir = DATASET_ROOT / "annotations"
    ann_dir.mkdir(exist_ok=True)

    for split in SPLITS:
        coco = build_coco_json(split)
        out_path = ann_dir / f"instances_{split}.json"
        with open(out_path, "w") as f:
            json.dump(coco, f)
        print(f"[{split}] saved → {out_path}  "
              f"({len(coco['images'])} images, {len(coco['annotations'])} annotations)")


if __name__ == "__main__":
    main()
