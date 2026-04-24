#!/usr/bin/env python3
"""
Compare COCO bbox eval metrics between two training logs.

Expected log format:
- one JSON object per line
- contains keys: "epoch" and "test_coco_eval_bbox"
- test_coco_eval_bbox is a list with 12 values:
  [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
]


def read_json_lines(log_path: Path) -> List[dict]:
    rows: List[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_bbox_by_epoch(rows: List[dict]) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for row in rows:
        epoch = row.get("epoch")
        bbox = row.get("test_coco_eval_bbox")
        if not isinstance(epoch, int):
            continue
        if not isinstance(bbox, list) or len(bbox) < len(METRIC_NAMES):
            continue
        out[epoch] = [float(v) for v in bbox[: len(METRIC_NAMES)]]
    return out


def metric_str(v: float) -> str:
    return f"{v:.6f}"


def build_metric_table(left: List[float], right: List[float], left_name: str, right_name: str) -> str:
    lines = []
    header = f"{'Metric':<8}  {left_name:>12}  {right_name:>12}  {'Delta(B-A)':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, metric in enumerate(METRIC_NAMES):
        a = left[i]
        b = right[i]
        d = b - a
        lines.append(f"{metric:<8}  {metric_str(a):>12}  {metric_str(b):>12}  {metric_str(d):>12}")
    return "\n".join(lines)


def best_epoch(metrics_by_epoch: Dict[int, List[float]], metric_index: int = 0) -> Optional[int]:
    if not metrics_by_epoch:
        return None
    return max(metrics_by_epoch.keys(), key=lambda e: metrics_by_epoch[e][metric_index])


def summarize_run(name: str, metrics_by_epoch: Dict[int, List[float]]) -> str:
    if not metrics_by_epoch:
        return f"{name}: no valid bbox eval rows found"

    epochs = sorted(metrics_by_epoch.keys())
    final_epoch = epochs[-1]
    best_ap_epoch = best_epoch(metrics_by_epoch, metric_index=0)

    assert best_ap_epoch is not None
    final_ap = metrics_by_epoch[final_epoch][0]
    best_ap = metrics_by_epoch[best_ap_epoch][0]

    return (
        f"{name}: eval_epochs={len(epochs)}, first={epochs[0]}, last={final_epoch}, "
        f"final_AP={final_ap:.6f}, best_AP={best_ap:.6f} @ epoch {best_ap_epoch}"
    )


def save_common_csv(
    csv_path: Path,
    common_epochs: List[int],
    a_by_epoch: Dict[int, List[float]],
    b_by_epoch: Dict[int, List[float]],
    a_name: str,
    b_name: str,
) -> None:
    fieldnames = ["epoch"]
    for m in METRIC_NAMES:
        fieldnames.extend([f"{m}_{a_name}", f"{m}_{b_name}", f"{m}_delta_{b_name}-minus-{a_name}"])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in common_epochs:
            row = {"epoch": epoch}
            a_vals = a_by_epoch[epoch]
            b_vals = b_by_epoch[epoch]
            for i, m in enumerate(METRIC_NAMES):
                row[f"{m}_{a_name}"] = a_vals[i]
                row[f"{m}_{b_name}"] = b_vals[i]
                row[f"{m}_delta_{b_name}-minus-{a_name}"] = b_vals[i] - a_vals[i]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare COCO bbox eval metrics between two log files")
    parser.add_argument("--log-a", type=Path, required=True, help="Path to first run log.txt")
    parser.add_argument("--log-b", type=Path, required=True, help="Path to second run log.txt")
    parser.add_argument("--name-a", type=str, default="runA", help="Display name for log-a")
    parser.add_argument("--name-b", type=str, default="runB", help="Display name for log-b")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Specific epoch to compare. If omitted, uses latest common epoch.",
    )
    parser.add_argument(
        "--show-common-ap",
        action="store_true",
        help="Also print AP/AP50/AP75 deltas over all common epochs.",
    )
    parser.add_argument(
        "--save-common-csv",
        type=Path,
        default=None,
        help="Optional path to save per-epoch comparison over common epochs as CSV.",
    )
    args = parser.parse_args()

    rows_a = read_json_lines(args.log_a)
    rows_b = read_json_lines(args.log_b)

    a_by_epoch = extract_bbox_by_epoch(rows_a)
    b_by_epoch = extract_bbox_by_epoch(rows_b)

    print("=== Run Summaries ===")
    print(summarize_run(args.name_a, a_by_epoch))
    print(summarize_run(args.name_b, b_by_epoch))

    if not a_by_epoch or not b_by_epoch:
        raise SystemExit("One or both logs have no valid bbox eval rows.")

    common_epochs = sorted(set(a_by_epoch.keys()) & set(b_by_epoch.keys()))
    if not common_epochs:
        raise SystemExit("No common epochs found between the two logs.")

    if args.epoch is not None:
        compare_epoch = args.epoch
        if compare_epoch not in a_by_epoch or compare_epoch not in b_by_epoch:
            raise SystemExit(f"Epoch {compare_epoch} is not available in both logs.")
    else:
        compare_epoch = common_epochs[-1]

    print("\n=== Detailed Comparison ===")
    print(f"Epoch compared: {compare_epoch}")
    print(build_metric_table(a_by_epoch[compare_epoch], b_by_epoch[compare_epoch], args.name_a, args.name_b))

    best_a_epoch = best_epoch(a_by_epoch, metric_index=0)
    best_b_epoch = best_epoch(b_by_epoch, metric_index=0)

    if best_a_epoch is not None and best_b_epoch is not None:
        print("\n=== Best AP Epochs ===")
        print(f"{args.name_a}: AP={a_by_epoch[best_a_epoch][0]:.6f} @ epoch {best_a_epoch}")
        print(f"{args.name_b}: AP={b_by_epoch[best_b_epoch][0]:.6f} @ epoch {best_b_epoch}")

    if args.show_common_ap:
        print("\n=== Common Epoch AP/AP50/AP75 Deltas (B-A) ===")
        print(f"{'Epoch':>6}  {'AP':>10}  {'AP50':>10}  {'AP75':>10}")
        print("-" * 44)
        for epoch in common_epochs:
            da = b_by_epoch[epoch][0] - a_by_epoch[epoch][0]
            d50 = b_by_epoch[epoch][1] - a_by_epoch[epoch][1]
            d75 = b_by_epoch[epoch][2] - a_by_epoch[epoch][2]
            print(f"{epoch:>6}  {da:>10.6f}  {d50:>10.6f}  {d75:>10.6f}")

    if args.save_common_csv is not None:
        save_common_csv(
            csv_path=args.save_common_csv,
            common_epochs=common_epochs,
            a_by_epoch=a_by_epoch,
            b_by_epoch=b_by_epoch,
            a_name=args.name_a,
            b_name=args.name_b,
        )
        print(f"\nSaved common-epoch CSV: {args.save_common_csv}")


if __name__ == "__main__":
    main()
