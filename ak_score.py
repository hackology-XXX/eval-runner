# ABOUTME: COCO→AK conversion layer wrapping the vendored AK metric (assecobs_quality_metrics).
# ABOUTME: Provides compute_ak(), compute_ak_from_files(), and a CLI entry point.
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Layout-agnostic import: package path in this repo, flat path when copied
# into eval-runner at deploy time (sibling ak_vendor/ directory).
try:
    from hackology.ak_vendor import (
        AvgWeightMode,
        BoundingBox,
        GroundTruthPredictionsPair,
        PredictionBoundingBox,
        calculate_ak,
    )
except ImportError:  # pragma: no cover - exercised only in the flat eval-runner layout
    from ak_vendor import (  # type: ignore[no-redef]
        AvgWeightMode,
        BoundingBox,
        GroundTruthPredictionsPair,
        PredictionBoundingBox,
        calculate_ak,
    )


@dataclass(frozen=True)
class AKResult:
    """Result of AK computation."""

    ak: float  # weighted-average AK across classes, in [0, 1]
    n_predictions: int
    n_ground_truth: int  # raw annotation count, includes any degenerate ones in n_skipped_boxes
    n_skipped_boxes: int  # degenerate boxes (w<=0/h<=0, non-finite, malformed) dropped during conversion
    n_below_threshold: int = 0  # predictions dropped for score < score_threshold


def _is_degenerate(bbox: list[float]) -> bool:
    """COCO bbox is [x, y, w, h]; degenerate when it cannot form a valid corner box.

    Drops zero/negative width or height (left>=right / top>=bottom would raise in
    BoundingBox) and any non-finite coordinate (NaN/Inf compare False against every
    threshold, so they would silently slip past validation into the metric).
    """
    if len(bbox) != 4 or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox
    ):
        return True
    if not all(math.isfinite(v) for v in bbox):
        return True
    _x, _y, w, h = bbox
    return w <= 0 or h <= 0


def compute_ak(
    predictions: list[dict],
    ground_truth_coco: dict,
    score_threshold: float = 0.5,
) -> AKResult:
    """Compute the AK quality metric for COCO predictions against a COCO GT dict.

    AK = TP / (TP + FN + FP) per class, weighted-averaged across classes
    (weight = TP + FN + FP), matched at IoU >= 0.5. Mirrors the production
    semantics of assecobs_quality_metrics.calculate_ak (default weight mode
    TP_FN_FP, is_extra_pixel_in_iou=True).

    Unlike mAP (which integrates a precision-recall curve), AK counts every
    prediction as TP/FP with no confidence weighting, so unfiltered
    over-prediction tanks the score. Predictions with score < score_threshold
    are therefore dropped first (default 0.5). A prediction with no score field
    is kept (cannot be thresholded). Set score_threshold=0.0 to disable.

    Degenerate boxes (w<=0 or h<=0, non-finite, malformed) are dropped before
    evaluation — the vendored BoundingBox would otherwise raise ValueError.
    Drop counts are returned for transparency.

    Args:
        predictions: List of dicts with image_id, category_id, bbox [x,y,w,h], score.
        ground_truth_coco: COCO dict with images, annotations, categories.
        score_threshold: Minimum prediction score to keep (default 0.5).

    Returns:
        AKResult with the weighted AK, prediction/GT counts, and drop counts.
    """
    n_predictions = len(predictions)
    n_ground_truth = len(ground_truth_coco.get("annotations", []))
    skipped = 0

    # Score threshold: drop low-confidence predictions (None score is unthresholdable → kept).
    below = 0
    if score_threshold > 0.0:
        kept = []
        for p in predictions:
            s = p.get("score")
            if s is not None and s < score_threshold:
                below += 1
            else:
                kept.append(p)
        predictions = kept

    # Image universe: every GT image plus any image referenced by predictions.
    image_ids = {img["id"] for img in ground_truth_coco.get("images", [])}
    image_ids.update(p["image_id"] for p in predictions)

    gt_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in ground_truth_coco.get("annotations", []):
        gt_by_image[ann["image_id"]].append(ann)

    preds_by_image: dict[int, list[dict]] = defaultdict(list)
    for pred in predictions:
        preds_by_image[pred["image_id"]].append(pred)

    pairs: list[GroundTruthPredictionsPair] = []
    for image_id in image_ids:
        gt_boxes: list[BoundingBox] = []
        for ann in gt_by_image.get(image_id, []):
            if _is_degenerate(ann["bbox"]):
                skipped += 1
                continue
            x, y, w, h = ann["bbox"]
            gt_boxes.append(
                BoundingBox(
                    left=x, top=y, right=x + w, bottom=y + h,
                    label=str(ann["category_id"]), confidence=None, loose=False,
                )
            )

        pred_boxes: list[PredictionBoundingBox] = []
        for pred in preds_by_image.get(image_id, []):
            if _is_degenerate(pred["bbox"]):
                skipped += 1
                continue
            x, y, w, h = pred["bbox"]
            pred_boxes.append(
                PredictionBoundingBox(
                    left=x, top=y, right=x + w, bottom=y + h,
                    label=str(pred["category_id"]),
                    confidence=pred.get("score"), loose=False,
                )
            )

        pairs.append(
            GroundTruthPredictionsPair(
                ground_truth=gt_boxes, predictions=pred_boxes, image_id=str(image_id)
            )
        )

    ak = calculate_ak(
        pairs,
        ak_weight_mode=AvgWeightMode.TP_FN_FP,
        min_overlap=0.5,
        is_extra_pixel_in_iou=True,  # explicit: match upstream default, guard re-vendor drift
    )
    # calc_global_weighted_value returns 0 (int) when total weight is 0.
    ak = float(ak) if ak is not None else 0.0

    return AKResult(
        ak=ak,
        n_predictions=n_predictions,
        n_ground_truth=n_ground_truth,
        n_skipped_boxes=skipped,
        n_below_threshold=below,
    )


def safe_compute_ak(predictions: list[dict], ground_truth_coco: dict) -> AKResult | None:
    """Best-effort AK: return None instead of raising.

    AK is informational and must never drop a valid mAP score. Any failure
    (malformed input, OOM, a future vendored-code bug) is swallowed, logged to
    stderr as a GitHub Actions warning annotation with a traceback, and surfaced
    as None so the caller can record mAP regardless.
    """
    try:
        return compute_ak(predictions, ground_truth_coco)
    except Exception as e:  # noqa: BLE001 - AK is best-effort; mAP scoring is the gate
        print(
            f"::warning::AK computation failed, recording mAP only: {e}\n"
            f"{traceback.format_exc()}",
            file=sys.stderr,
        )
        return None


def compute_ak_from_files(
    predictions_path: Path, gt_path: Path, score_threshold: float = 0.5
) -> AKResult:
    """Compute AK from JSON files on disk.

    Args:
        predictions_path: Path to predictions JSON (list of dicts).
        gt_path: Path to COCO ground-truth JSON.
        score_threshold: Minimum prediction score to keep (default 0.5).
    """
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    ground_truth = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    return compute_ak(predictions, ground_truth, score_threshold=score_threshold)


def main() -> None:
    """CLI entry point: score-ak --predictions PATH --gt PATH [--score-threshold T]"""
    parser = argparse.ArgumentParser(description="Compute AK quality metric for COCO predictions")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSON")
    parser.add_argument("--gt", type=Path, required=True, help="Path to ground-truth COCO JSON")
    parser.add_argument("--score-threshold", type=float, default=0.5,
                        help="Minimum prediction score to keep (default 0.5; 0 disables)")
    args = parser.parse_args()

    result = compute_ak_from_files(args.predictions, args.gt, score_threshold=args.score_threshold)
    print(f"AK@{args.score_threshold:g}:      {result.ak:.4f}")
    print(f"Predictions:  {result.n_predictions}")
    print(f"Ground truth: {result.n_ground_truth}")
    if result.n_below_threshold:
        print(f"Dropped (score < {args.score_threshold:g}): {result.n_below_threshold}")
    if result.n_skipped_boxes:
        print(f"Skipped boxes (degenerate): {result.n_skipped_boxes}")
    sys.exit(0)


if __name__ == "__main__":
    main()
