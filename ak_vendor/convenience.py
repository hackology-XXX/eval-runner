# ABOUTME: High-level AK entry point (weight-mode selection + weighted-average AK).
# ABOUTME: Vendored subset of assecobs_quality_metrics @ 7fa3f25 (2026-06-03) — DO NOT EDIT LOGIC.
from typing import Callable, List

from .evaluation import (
    aggregate_evaluation_results,
    calc_evaluation_metrics_per_key,
    calc_global_weighted_value,
    evaluate_boxes,
    filter_loose_results,
)
from .exceptions import (
    InvalidWeightModeException,
)
from .od_types import (
    AvgWeightMode,
)
from .types import (
    ImageEvaluatedBox,
)


def weight_fn(ak_weight_mode):
    if ak_weight_mode == AvgWeightMode.GROUND_TRUTH:
        return lambda m: m.TP + m.FN
    if ak_weight_mode == AvgWeightMode.GROUND_TRUTH_AND_PREDICTIONS:
        return lambda m: m.TP + m.FN + m.TP + m.FP
    if ak_weight_mode == AvgWeightMode.TP_FN_FP:
        return lambda m: m.TP + m.FN + m.FP
    if ak_weight_mode == AvgWeightMode.EQUAL:
        return lambda m: 1
    if ak_weight_mode == AvgWeightMode.IMAGES_COUNT:
        return lambda m: m.images_count

    raise InvalidWeightModeException(f"invalid weight mode {ak_weight_mode}")


def calculate_global_from_image_evaluated_boxes(
    value_fn: Callable,
    image_evaluated_boxes: List[ImageEvaluatedBox],
    ak_weight_mode=AvgWeightMode.TP_FN_FP,
    excluded_classes=None,
    ak_precision=None,
):
    if excluded_classes is None:
        excluded_classes = []
    image_evaluated_boxes = filter_loose_results(image_evaluated_boxes)
    class_aggregate_evaluation_results = aggregate_evaluation_results(
        image_evaluated_boxes, "class"
    )
    if len(excluded_classes) > 0:
        class_aggregate_evaluation_results = {
            c: r
            for c, r in class_aggregate_evaluation_results.items()
            if c not in excluded_classes
        }

    class_evaluation_metrics = calc_evaluation_metrics_per_key(
        class_aggregate_evaluation_results
    )

    value = calc_global_weighted_value(
        value_fn, class_evaluation_metrics, weight_fn(ak_weight_mode), ak_precision
    )
    return value


def calculate_ak(
    ground_truth_predictions_pairs,
    ak_weight_mode=AvgWeightMode.TP_FN_FP,
    only_gt_classes=False,
    min_overlap=0.5,
    excluded_classes=None,
    ak_precision=None,
    is_extra_pixel_in_iou=True,
):
    # pylint: disable=too-many-arguments,invalid-name
    if excluded_classes is None:
        excluded_classes = []
    image_evaluated_boxes = evaluate_boxes(
        ground_truth_predictions_pairs,
        min_overlap,
        only_gt_classes,
        is_extra_pixel_in_iou,
    )
    ak = calculate_global_from_image_evaluated_boxes(
        lambda m: m.AK,
        image_evaluated_boxes,
        ak_weight_mode,
        excluded_classes,
        ak_precision,
    )
    return ak
