"""
Compute YOLO-style threshold metrics (P, R, F1 at best operating point)
from DEIMv2 validation evaluation outputs.

This script can:
1) run validation from a checkpoint (producing output_dir/eval.pth), and
2) parse COCO eval tensors to report P/R/F1 at IoU=0.5, area=all, maxDets=100.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from engine.misc import dist_utils
from engine.core import YAMLConfig
from engine.solver import TASKS


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _safe_float(v: float) -> float:
    return float(v) if np.isfinite(v) else float("nan")


def run_validation(config_path: str, checkpoint_path: str, device: str, seed: int) -> Dict[str, Any]:
    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=seed)

    cfg = YAMLConfig(config_path, resume=checkpoint_path, device=device)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.val()

    eval_pth = Path(cfg.output_dir) / "eval.pth"
    if not eval_pth.exists():
        raise FileNotFoundError(f"Validation finished but eval file not found: {eval_pth}")

    dist_utils.cleanup()
    return {
        "eval_pth": str(eval_pth),
        "output_dir": str(cfg.output_dir),
        "ann_file": cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"],
    }


def load_category_names(ann_file: str) -> Dict[int, str]:
    try:
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {int(c["id"]): str(c.get("name", c["id"])) for c in data.get("categories", [])}
    except Exception:
        return {}


def compute_yolo_style_metrics(eval_dict: Dict[str, Any], iou: float, area: str, max_dets: int) -> Dict[str, Any]:
    precision = np.asarray(eval_dict["precision"])  # [T, R, K, A, M]
    scores = np.asarray(eval_dict["scores"])        # [T, R, K, A, M]

    params = eval_dict["params"]
    iou_thrs = np.asarray(params.iouThrs)
    rec_thrs = np.asarray(params.recThrs)
    area_labels: List[str] = list(params.areaRngLbl)
    max_dets_list: List[int] = list(params.maxDets)
    cat_ids: List[int] = [int(x) for x in list(params.catIds)]

    t_idx = _nearest_index(iou_thrs, iou)
    if area not in area_labels:
        raise ValueError(f"Area label '{area}' not in {area_labels}")
    a_idx = area_labels.index(area)
    m_idx = _nearest_index(np.asarray(max_dets_list), float(max_dets))

    p_slice = precision[t_idx, :, :, a_idx, m_idx]  # [R, K]
    s_slice = scores[t_idx, :, :, a_idx, m_idx]     # [R, K]

    p_curve = []
    score_curve = []
    for r in range(p_slice.shape[0]):
        p_row = p_slice[r]
        s_row = s_slice[r]
        valid = p_row > -1
        if np.any(valid):
            p_curve.append(float(np.mean(p_row[valid])))
            score_curve.append(float(np.mean(s_row[valid])))
        else:
            p_curve.append(float("nan"))
            score_curve.append(float("nan"))

    p_curve = np.asarray(p_curve)
    score_curve = np.asarray(score_curve)
    r_curve = rec_thrs

    f1_curve = 2.0 * p_curve * r_curve / (p_curve + r_curve + 1e-16)
    if np.all(np.isnan(f1_curve)):
        raise RuntimeError("All F1 values are NaN. Check eval tensors and selected area/maxDets.")

    best_idx = int(np.nanargmax(f1_curve))

    per_class = []
    for k in range(p_slice.shape[1]):
        p_k = p_slice[:, k]
        s_k = s_slice[:, k]
        valid = p_k > -1
        if not np.any(valid):
            continue

        p_k_nan = np.where(valid, p_k, np.nan)
        s_k_nan = np.where(valid, s_k, np.nan)
        f1_k = 2.0 * p_k_nan * r_curve / (p_k_nan + r_curve + 1e-16)
        best_k = int(np.nanargmax(f1_k))

        per_class.append(
            {
                "category_id": int(cat_ids[k]) if k < len(cat_ids) else int(k),
                "precision": _safe_float(p_k_nan[best_k]),
                "recall": _safe_float(r_curve[best_k]),
                "f1": _safe_float(f1_k[best_k]),
                "score_threshold": _safe_float(s_k_nan[best_k]),
                "best_recall_index": best_k,
            }
        )

    return {
        "selection": {
            "iou_requested": iou,
            "iou_used": float(iou_thrs[t_idx]),
            "area_requested": area,
            "area_used": area_labels[a_idx],
            "max_dets_requested": max_dets,
            "max_dets_used": int(max_dets_list[m_idx]),
        },
        "overall": {
            "precision": _safe_float(p_curve[best_idx]),
            "recall": _safe_float(r_curve[best_idx]),
            "f1": _safe_float(f1_curve[best_idx]),
            "score_threshold": _safe_float(score_curve[best_idx]),
            "best_recall_index": best_idx,
        },
        "curves": {
            "recall": r_curve.tolist(),
            "precision": p_curve.tolist(),
            "f1": f1_curve.tolist(),
            "score_threshold": score_curve.tolist(),
        },
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/deimv2/deimv2_hgnetv2_n_fisheye.yml")
    parser.add_argument("-r", "--resume", type=str, default="outputs/deimv2_hgnetv2_n_fisheye/best_stg1.pth")
    parser.add_argument("-d", "--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--run-val", action="store_true", help="Run validation before parsing metrics")
    parser.add_argument("--eval-pth", type=str, default="", help="Path to eval.pth; if set, skip validation run")

    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--area", type=str, default="all", choices=["all", "small", "medium", "large"])
    parser.add_argument("--max-dets", type=int, default=100)

    parser.add_argument("--save-json", type=str, default="")
    args = parser.parse_args()

    cfg = YAMLConfig(args.config, resume=args.resume, device=args.device)
    ann_file = cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"]
    output_dir = Path(cfg.output_dir)

    if args.eval_pth:
        eval_pth = Path(args.eval_pth)
    else:
        eval_pth = output_dir / "eval.pth"
        if args.run_val or not eval_pth.exists():
            run_info = run_validation(args.config, args.resume, args.device, args.seed)
            eval_pth = Path(run_info["eval_pth"])
            ann_file = run_info["ann_file"]

    if not eval_pth.exists():
        raise FileNotFoundError(
            f"Could not find eval file: {eval_pth}. Use --run-val or provide --eval-pth explicitly."
        )

    eval_dict = torch.load(str(eval_pth), map_location="cpu")
    result = compute_yolo_style_metrics(eval_dict, iou=args.iou, area=args.area, max_dets=args.max_dets)

    id2name = load_category_names(ann_file)
    for item in result["per_class"]:
        item["category_name"] = id2name.get(item["category_id"], str(item["category_id"]))

    print("=== YOLO-style Threshold Metrics (from COCO eval tensors) ===")
    print(f"eval_pth: {eval_pth}")
    print(f"selection: {result['selection']}")
    print(
        "overall: "
        f"P={result['overall']['precision']:.6f}, "
        f"R={result['overall']['recall']:.6f}, "
        f"F1={result['overall']['f1']:.6f}, "
        f"score={result['overall']['score_threshold']:.6f}"
    )

    if result["per_class"]:
        print("per-class best-F1 metrics:")
        for item in result["per_class"]:
            print(
                f"  cat={item['category_id']} ({item['category_name']}): "
                f"P={item['precision']:.6f}, R={item['recall']:.6f}, "
                f"F1={item['f1']:.6f}, score={item['score_threshold']:.6f}"
            )

    if args.save_json:
        save_path = Path(args.save_json)
    else:
        stem = Path(args.resume).stem if args.resume else "model"
        save_path = output_dir / f"yolo_precision_{stem}.json"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"saved_json: {save_path}")


if __name__ == "__main__":
    main()
