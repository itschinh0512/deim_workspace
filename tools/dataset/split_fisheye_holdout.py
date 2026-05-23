"""Create a fresh train/val split from the fisheye train annotations.

This keeps the existing val annotations untouched so they can continue to serve
as the final held-out test set.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path


def build_split_annotations(source_annotations: dict, split_image_ids: set[int]) -> dict:
    images = [image for image in source_annotations["images"] if image["id"] in split_image_ids]
    annotations = [annotation for annotation in source_annotations["annotations"] if annotation["image_id"] in split_image_ids]

    return {
        "info": deepcopy(source_annotations.get("info", {})),
        "licenses": deepcopy(source_annotations.get("licenses", [])),
        "categories": deepcopy(source_annotations["categories"]),
        "images": images,
        "annotations": annotations,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new fisheye train/val split.")
    parser.add_argument(
        "--train-ann",
        type=Path,
        default=Path("/home/locth/omni2rect_DEIM/dataset_fisheye/annotations/instances_train.json"),
        help="Source train COCO annotation file.",
    )
    parser.add_argument(
        "--val-ann",
        type=Path,
        default=Path("/home/locth/omni2rect_DEIM/dataset_fisheye/annotations/instances_val.json"),
        help="Existing validation COCO annotation file to preserve as test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/locth/omni2rect_DEIM/dataset_fisheye/annotations/splits"),
        help="Directory where split annotation files will be written.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Fraction of the original train set to move into the new validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.train_ann.open("r", encoding="utf-8") as handle:
        train_annotations = json.load(handle)

    with args.val_ann.open("r", encoding="utf-8") as handle:
        val_annotations = json.load(handle)

    train_image_ids = [image["id"] for image in train_annotations["images"]]
    if not train_image_ids:
        raise SystemExit(f"No train images found in {args.train_ann}")

    val_size = max(1, round(len(train_image_ids) * args.val_fraction))
    random.Random(args.seed).shuffle(train_image_ids)
    new_val_image_ids = set(train_image_ids[:val_size])
    new_train_image_ids = set(train_image_ids[val_size:])

    new_train_annotations = build_split_annotations(train_annotations, new_train_image_ids)
    new_val_annotations = build_split_annotations(train_annotations, new_val_image_ids)

    test_annotations = deepcopy(val_annotations)
    test_annotations["info"] = deepcopy(test_annotations.get("info", {}))
    test_annotations["info"]["description"] = "Fisheye license plate dataset - test"

    write_json(args.output_dir / "instances_train_split.json", new_train_annotations)
    write_json(args.output_dir / "instances_val_split.json", new_val_annotations)
    write_json(args.output_dir / "instances_test.json", test_annotations)

    print(f"Wrote {len(new_train_annotations['images'])} train images to instances_train_split.json")
    print(f"Wrote {len(new_val_annotations['images'])} val images to instances_val_split.json")
    print(f"Kept {len(test_annotations['images'])} images in instances_test.json")


if __name__ == "__main__":
    main()