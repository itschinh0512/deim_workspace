"""
Batch torch inference on validation images for DEIM/DEIMv2 models.

This script runs one or more checkpoints on a validation image directory,
saves rendered predictions for each image, and exports JSONL prediction logs.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from engine.core import YAMLConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch torch inference on a validation image folder")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "-r",
        "--resume",
        nargs="+",
        type=str,
        required=True,
        help="One or more model checkpoint paths (.pth)",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=None,
        help="Validation image directory. If omitted, use val_dataloader.dataset.img_folder in config.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory. Default: ../inference_results (from deim_workspace)",
    )
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--score-threshold", type=float, default=0.45, help="Score threshold for drawing/saving detections")
    parser.add_argument("--max-images", type=int, default=0, help="Max number of images to run. 0 means all.")
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="",
        help="Optional prefix for each run folder name",
    )
    return parser.parse_args()


def load_checkpoint_state(checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint and "module" in checkpoint["ema"]:
        return checkpoint["ema"]["module"]
    if "model" in checkpoint:
        return checkpoint["model"]
    raise KeyError(f"Cannot find model weights in checkpoint: {checkpoint_path}")


def resolve_input_dir(cfg: YAMLConfig, input_dir: str = None) -> Path:
    if input_dir:
        return Path(input_dir).expanduser().resolve()

    try:
        folder = cfg.yaml_cfg["val_dataloader"]["dataset"]["img_folder"]
    except KeyError as exc:
        raise KeyError("input-dir not provided and val_dataloader.dataset.img_folder missing in config") from exc

    return Path(folder).expanduser().resolve()


def build_transforms(size: Sequence[int], vit_backbone: bool) -> T.Compose:
    return T.Compose(
        [
            T.Resize(tuple(size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if vit_backbone else T.Lambda(lambda x: x),
        ]
    )


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def output_to_tensors(output) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Deploy mode: tuple(labels, boxes, scores)
    if isinstance(output, tuple) and len(output) == 3:
        return output

    # Non-deploy mode: list[dict(labels, boxes, scores)]
    if isinstance(output, list) and output and isinstance(output[0], dict):
        item = output[0]
        labels = item["labels"].unsqueeze(0)
        boxes = item["boxes"].unsqueeze(0)
        scores = item["scores"].unsqueeze(0)
        return labels, boxes, scores

    raise TypeError(f"Unsupported model output type: {type(output)}")


def draw_predictions(image: Image.Image, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 69, 0), width=2)
        draw.text((x1, max(y1 - 12, 0)), f"cls_{int(label)} {float(score):.2f}", fill=(0, 191, 255))

    return canvas


def build_model(cfg: YAMLConfig, checkpoint_path: str, device: torch.device) -> nn.Module:
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    state = load_checkpoint_state(checkpoint_path)
    cfg.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = DeployModel().to(device)
    model.eval()
    return model


def run_single_checkpoint(
    config_path: str,
    checkpoint_path: str,
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
    score_threshold: float,
    max_images: int,
) -> None:
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    model = build_model(cfg, checkpoint_path, device)

    img_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    vit_backbone = bool(cfg.yaml_cfg.get("DINOv3STAs", False))
    transforms = build_transforms(img_size, vit_backbone)

    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    pred_log_path = output_dir / "predictions.jsonl"

    image_paths: List[Path] = list(iter_images(input_dir))
    if max_images > 0:
        image_paths = image_paths[:max_images]

    print(f"[{checkpoint_path}] Found {len(image_paths)} validation images")

    with torch.no_grad(), pred_log_path.open("w", encoding="utf-8") as log_f:
        iterator = (
            tqdm(
                image_paths,
                total=len(image_paths),
                desc=f"Infer {Path(checkpoint_path).name}",
                unit="img",
                dynamic_ncols=True,
            )
            if tqdm is not None
            else image_paths
        )

        for idx, image_path in enumerate(iterator, start=1):
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            orig_size = torch.tensor([[width, height]], dtype=torch.float32, device=device)
            tensor = transforms(image).unsqueeze(0).to(device)

            output = model(tensor, orig_size)
            labels, boxes, scores = output_to_tensors(output)

            labels = labels[0].detach().cpu()
            boxes = boxes[0].detach().cpu()
            scores = scores[0].detach().cpu()

            keep = scores >= score_threshold
            labels_k = labels[keep]
            boxes_k = boxes[keep]
            scores_k = scores[keep]

            rel = image_path.relative_to(input_dir)
            save_path = images_out / rel
            save_path.parent.mkdir(parents=True, exist_ok=True)

            rendered = draw_predictions(image, boxes_k, labels_k, scores_k)
            rendered.save(save_path)

            record = {
                "image_path": str(image_path),
                "output_image": str(save_path),
                "image_size": [width, height],
                "num_detections": int(scores_k.numel()),
                "detections": [
                    {
                        "label": int(label.item()),
                        "score": float(score.item()),
                        "bbox_xyxy": [float(v) for v in box.tolist()],
                    }
                    for label, score, box in zip(labels_k, scores_k, boxes_k)
                ],
            }
            log_f.write(json.dumps(record, ensure_ascii=True) + "\n")

            if tqdm is None and (idx == 1 or idx % 100 == 0 or idx == len(image_paths)):
                print(f"[{checkpoint_path}] Processed {idx}/{len(image_paths)}")

    run_info = {
        "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "score_threshold": score_threshold,
        "max_images": max_images,
        "num_images": len(image_paths),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with (output_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=True)


def build_run_dir(base_output_dir: Path, run_prefix: str, checkpoint_path: str) -> Path:
    ckpt_stem = Path(checkpoint_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_prefix:
        run_name = f"{run_prefix}_{ckpt_stem}_{timestamp}"
    else:
        run_name = f"{ckpt_stem}_{timestamp}"
    return base_output_dir / run_name


def main() -> None:
    args = parse_args()
    workspace_dir = Path(__file__).resolve().parents[2]
    default_output = workspace_dir.parent / "inference_results"
    base_output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output.resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    cfg_for_path = YAMLConfig(args.config, resume=args.resume[0])
    input_dir = resolve_input_dir(cfg_for_path, args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Validation input directory does not exist: {input_dir}")

    for checkpoint_path in args.resume:
        run_dir = build_run_dir(base_output_dir, args.run_prefix, checkpoint_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_single_checkpoint(
            config_path=args.config,
            checkpoint_path=checkpoint_path,
            input_dir=input_dir,
            output_dir=run_dir,
            device=device,
            score_threshold=args.score_threshold,
            max_images=args.max_images,
        )
        print(f"Saved run outputs to: {run_dir}")


if __name__ == "__main__":
    main()
