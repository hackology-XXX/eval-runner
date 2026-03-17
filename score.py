# ABOUTME: Wrapper around pycocotools COCOeval for computing mAP scores.
# ABOUTME: Provides compute_map(), compute_map_from_files(), and a CLI entry point.
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@dataclass(frozen=True)
class ScoreResult:
    """Result of mAP computation."""

    map_at_05: float  # AP @ IoU=0.5
    map_at_05_095: float  # AP @ IoU=0.50:0.95
    n_predictions: int
    n_ground_truth: int


def compute_map(predictions: list[dict], ground_truth_coco: dict) -> ScoreResult:
    """Compute mAP scores for predictions against a COCO ground-truth dict.

    Args:
        predictions: List of prediction dicts with image_id, category_id, bbox, score.
        ground_truth_coco: COCO-format dict with images, annotations, categories.

    Returns:
        ScoreResult with mAP@0.5, mAP@0.5:0.95, and counts.
    """
    n_gt = len(ground_truth_coco.get("annotations", []))

    if not predictions:
        return ScoreResult(
            map_at_05=0.0,
            map_at_05_095=0.0,
            n_predictions=0,
            n_ground_truth=n_gt,
        )

    # Suppress pycocotools stdout spam
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = ground_truth_coco
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(predictions)

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    # stats[0] = AP @ IoU=0.50:0.95, stats[1] = AP @ IoU=0.50
    map_at_05_095 = float(coco_eval.stats[0])
    map_at_05 = float(coco_eval.stats[1])

    # pycocotools returns -1 when there are no valid detections
    if map_at_05 < 0:
        map_at_05 = 0.0
    if map_at_05_095 < 0:
        map_at_05_095 = 0.0

    return ScoreResult(
        map_at_05=map_at_05,
        map_at_05_095=map_at_05_095,
        n_predictions=len(predictions),
        n_ground_truth=n_gt,
    )


def compute_map_from_files(predictions_path: Path, gt_path: Path) -> ScoreResult:
    """Compute mAP from JSON files on disk.

    Args:
        predictions_path: Path to predictions JSON (list of dicts).
        gt_path: Path to COCO ground-truth JSON (dict with images/annotations/categories).
    """
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    ground_truth = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    return compute_map(predictions, ground_truth)


def main() -> None:
    """CLI entry point: score --predictions PATH --gt PATH"""
    parser = argparse.ArgumentParser(description="Compute mAP score for COCO predictions")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSON")
    parser.add_argument("--gt", type=Path, required=True, help="Path to ground-truth COCO JSON")
    args = parser.parse_args()

    result = compute_map_from_files(args.predictions, args.gt)
    print(f"mAP@0.5:      {result.map_at_05:.4f}")
    print(f"mAP@0.5:0.95: {result.map_at_05_095:.4f}")
    print(f"Predictions:   {result.n_predictions}")
    print(f"Ground truth:  {result.n_ground_truth}")
    sys.exit(0)
