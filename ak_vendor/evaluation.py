# ABOUTME: Box-level evaluation chain (IoU matching → per-class metrics → weighted AK).
# ABOUTME: Vendored from assecobs_quality_metrics @ 7fa3f25 (2026-06-03) — DO NOT EDIT LOGIC.
from copy import copy
from typing import Callable, Dict, List, Optional, Tuple, Union
from itertools import groupby
from functools import lru_cache
import math
from .exceptions import (
    InvalidAggregateKeyNameException,
)
from .od_types import (
    EvaluationMetrics,
    AggregatedEvaluationResults,
)
from .types import (
    GroundTruthPredictionsPair,
    ImageEvaluatedBox,
    ImageEvaluatedBoxWithIndex,
    GTBoundingBoxError,
    EvaluatedBox,
    EvaluatedBoxWithIndex,
    BoundingBox,
    EvaluationResult,
    PredictionBoundingBox,
)

# pylint: disable-next=pointless-string-statement
"""
Etapy liczenia metryk object detection(przykład użycia
    convenience.calculate_all_metrics):

1. evaluate_boxes
    - input: predykcje z opisami per obrazek
    - output: błędy per box

2. aggregate_evaluation_results
    - input: błędy per box
    - output: zagregowane błędy per wybrany poziom -
        typowo per klasa (inne opcje: obrazek, total)

3. calc_evaluation_metrics_per_key / calc_evaluation_metrics
    - input: zagregowane błędy (np. per klasa, ew. total)
    - output: wyliczone metryki (np. per klasa, ew. total)

4. calc_global_weighted_value
    - input: wyliczone metryki per poziom (klasa)
    - output: ważona średnia wybranej metryki, z uwzględnieniem zadanej funkcji ważącej
"""


def evaluate_boxes(
    image_evaluation_boxes_sets: List[GroundTruthPredictionsPair],
    min_overlap: float = 0.5,
    only_gt_classes: bool = False,
    is_extra_pixel_in_iou: bool = True,
) -> List[ImageEvaluatedBox]:
    indexed = evaluate_boxes_with_indices(
        image_evaluation_boxes_sets,
        min_overlap,
        only_gt_classes,
        is_extra_pixel_in_iou,
    )
    # EvaluatedBoxWithIndex dziedziczy z EvaluatedBox — przy konstrukcji
    # ImageEvaluatedBox wystarczy zrzutować przez konstruktor bazowy aby zwrócić
    # dokładnie typ udokumentowany w sygnaturze (List[ImageEvaluatedBox]).
    return [
        ImageEvaluatedBox(
            EvaluatedBox(
                ib.evaluated_box.bounding_box,
                ib.evaluated_box.evaluation_result,
                ib.evaluated_box.ground_truth_label,
                ib.evaluated_box.loose,
            ),
            ib.image_id,
        )
        for ib in indexed
    ]


def evaluate_boxes_with_indices(
    image_evaluation_boxes_sets: List[GroundTruthPredictionsPair],
    min_overlap: float = 0.5,
    only_gt_classes: bool = False,
    is_extra_pixel_in_iou: bool = True,
) -> List[ImageEvaluatedBoxWithIndex]:
    """Ewaluuje predykcje i eksponuje matched_gt_index per detekcja.

    Zachowuje identyczne przypisanie EvaluationResult co evaluate_boxes(),
    ale dodatkowo zwraca matched_gt_index — 0-based indeks do
    image_set.ground_truth GT bboxa z którym predykcja została sparowana
    po IoU. Semantyka pełna: patrz EvaluatedBoxWithIndex. Skrót:
    matched_gt_index jest None tylko dla EXTRA predykcji; dla MISSED
    wskazuje na sam GT (umożliwia korelację MISSED→GT slot).
    """
    image_evaluated_boxes: List[ImageEvaluatedBoxWithIndex] = []
    all_classes_from_gt = {
        bounding_box.label
        for image_set in image_evaluation_boxes_sets
        for bounding_box in image_set.ground_truth
    }

    for image_set in image_evaluation_boxes_sets:
        # Build (original_index, GTBoundingBoxError) tuples so we can carry
        # the per-image GT index through __evaluate_class_boxes and back.
        gt_indexed: List[Tuple[int, GTBoundingBoxError]] = [
            (
                i,
                GTBoundingBoxError(
                    bb.left,
                    bb.top,
                    bb.right,
                    bb.bottom,
                    bb.label,
                    bb.confidence,
                    bb.loose,
                    bb.face,
                ),
            )
            for i, bb in enumerate(image_set.ground_truth)
        ]

        classes_to_evaluate = list(
            set(
                [bb.label for bb in image_set.ground_truth]
                + [bb.label for bb in image_set.predictions]
            )
        )
        if only_gt_classes:
            classes_to_evaluate = [
                c for c in classes_to_evaluate if c in all_classes_from_gt
            ]

        for clazz in classes_to_evaluate:
            class_pred_bboxes = [
                bb for bb in image_set.predictions if bb.label == clazz
            ]

            class_evaluated_boxes = __evaluate_class_boxes(
                clazz,
                gt_indexed,
                class_pred_bboxes,
                min_overlap,
                is_extra_pixel_in_iou,
            )

            image_evaluated_boxes.extend(
                [
                    ImageEvaluatedBoxWithIndex(b, image_set.image_id)
                    for b in class_evaluated_boxes
                ]
            )

        for gt_idx, gt_bbox in gt_indexed:
            if not gt_bbox.has_prediction or not gt_bbox.has_match:
                image_evaluated_boxes.append(
                    ImageEvaluatedBoxWithIndex(
                        EvaluatedBoxWithIndex(
                            BoundingBox(
                                gt_bbox.left,
                                gt_bbox.top,
                                gt_bbox.right,
                                gt_bbox.bottom,
                                gt_bbox.label,
                                gt_bbox.confidence,
                                gt_bbox.loose,
                                gt_bbox.face,
                            ),
                            EvaluationResult.MISSED,
                            gt_bbox.label,
                            gt_bbox.loose,
                            gt_idx,
                        ),
                        image_set.image_id,
                    )
                )

    return image_evaluated_boxes


@lru_cache(maxsize=20)
def evaluate_boxes_with_cache(
    image_evaluation_boxes_sets: Tuple[GroundTruthPredictionsPair],
    min_overlap: float = 0.5,
    only_gt_classes: bool = False,
    is_extra_pixel_in_iou: bool = True,
):
    """Wersja cached evaluate_boxes (lru_cache maxsize=20).

    Zwraca List[ImageEvaluatedBox] BEZ pola matched_gt_index. Konsumenci
    potrzebujący indeksu sparowanego GT muszą wołać evaluate_boxes_with_indices
    bezpośrednio (bez cache). UWAGA: lru_cache zwraca identyczny obiekt listy
    przy każdym cache-hit — mutacja wyniku jest widoczna we wszystkich
    kolejnych wywołaniach z tym samym kluczem.
    """
    return evaluate_boxes(
        image_evaluation_boxes_sets,
        min_overlap,
        only_gt_classes,
        is_extra_pixel_in_iou,
    )


def __evaluate_class_boxes(
    clazz: str,
    gt_indexed: List[Tuple[int, GTBoundingBoxError]],
    class_pred_bboxes: List[PredictionBoundingBox],
    min_overlap: float,
    is_extra_pixel_in_iou: bool,
) -> List[EvaluatedBoxWithIndex]:
    extra_pixel = 1 if is_extra_pixel_in_iou else 0
    evaluated_boxes: List[EvaluatedBoxWithIndex] = []
    for pred_bbox in class_pred_bboxes:
        gt_match_idx, gt_match, iou_max = _find_match_based_on_iou(
            extra_pixel, gt_indexed, pred_bbox
        )

        # assign prediction as true positive/don't care/false positive
        if iou_max >= min_overlap:
            _evaluate_prediction_based_on_gt(
                clazz, evaluated_boxes, gt_match, gt_match_idx, pred_bbox
            )
        else:
            # false positive (extra bbox)
            evaluated_boxes.append(
                EvaluatedBoxWithIndex(
                    pred_bbox, EvaluationResult.EXTRA, None, pred_bbox.loose, None
                )
            )

    return evaluated_boxes


def _evaluate_prediction_based_on_gt(
    clazz: str,
    evaluated_boxes: List[EvaluatedBoxWithIndex],
    gt_match: GTBoundingBoxError,
    gt_match_idx: Optional[int],
    pred_bbox: PredictionBoundingBox,
) -> None:
    gt_match.has_prediction = True
    loose = pred_bbox.loose or gt_match.loose
    if gt_match.label == clazz:
        if not gt_match.has_match:
            # true positive
            gt_match.has_match = True
            pred_bbox = copy(pred_bbox)
            pred_bbox.face = pred_bbox.face and gt_match.face
            evaluated_boxes.append(
                EvaluatedBoxWithIndex(
                    pred_bbox, EvaluationResult.CORRECT, clazz, loose, gt_match_idx
                )
            )
        else:
            # false positive (multiple detection)
            evaluated_boxes.append(
                EvaluatedBoxWithIndex(
                    pred_bbox, EvaluationResult.DUPLICATE, None, loose, gt_match_idx
                )
            )
    else:
        # false positive (different class)
        evaluated_boxes.append(
            EvaluatedBoxWithIndex(
                pred_bbox,
                EvaluationResult.WRONG,
                gt_match.label,
                loose,
                gt_match_idx,
            )
        )


def _find_match_based_on_iou(
    extra_pixel: int,
    gt_indexed: List[Tuple[int, GTBoundingBoxError]],
    pred_bbox: PredictionBoundingBox,
    iou_max: float = -1,
) -> Tuple[Optional[int], Optional[GTBoundingBoxError], float]:
    gt_match: Optional[GTBoundingBoxError] = None
    gt_match_idx: Optional[int] = None
    for gt_idx, gt_bbox in gt_indexed:
        # optymalizacja - nie wywołujemy IOU gdy nie trzeba
        if (  # pylint: disable=too-many-boolean-expressions
            gt_bbox.right < pred_bbox.left
            or pred_bbox.right < pred_bbox.left
            or gt_bbox.left > pred_bbox.right
            or pred_bbox.left > gt_bbox.right
            or gt_bbox.bottom < pred_bbox.top
            or pred_bbox.bottom < pred_bbox.top
            or gt_bbox.top > pred_bbox.bottom
            or pred_bbox.top > gt_bbox.bottom
        ):
            continue

        iou = _iou(pred_bbox, gt_bbox, extra_pixel)
        if iou > iou_max:
            iou_max = iou
            gt_match = gt_bbox
            gt_match_idx = gt_idx
    return gt_match_idx, gt_match, iou_max


def _iou(first: BoundingBox, second: BoundingBox, extra_pixel):
    # pylint: disable=invalid-name
    bi = [
        # left 0, top 1, right 2, bottom 3
        first.left if first.left > second.left else second.left,
        first.top if first.top > second.top else second.top,
        first.right if first.right < second.right else second.right,
        first.bottom if first.bottom < second.bottom else second.bottom,
    ]
    iw = bi[2] - bi[0] + extra_pixel
    ih = bi[3] - bi[1] + extra_pixel
    if iw > 0 and ih > 0:
        # compute overlap (IoU) = area of intersection / area of union
        ua = (
            (first.right - first.left + extra_pixel)
            * (first.bottom - first.top + extra_pixel)
            + (second.right - second.left + extra_pixel)
            * (second.bottom - second.top + extra_pixel)
            - iw * ih
        )
        return iw * ih / ua

    return 0


def aggregate_evaluation_results(
    evaluated_boxes: List[ImageEvaluatedBox],
    aggregate_key_name: Optional[str],
) -> Union[AggregatedEvaluationResults, Dict[str, AggregatedEvaluationResults]]:
    stripped_evaluated_boxes = [
        {
            "image_id": ev.image_id,
            "class": ev.evaluated_box.bounding_box.label,
            "result": ev.evaluated_box.evaluation_result,
        }
        for ev in evaluated_boxes
    ]

    if aggregate_key_name is None:
        return __count_evaluation_results(stripped_evaluated_boxes)

    if aggregate_key_name in ["image_id", "class"]:
        aggregated_evaluation_results = {}

        stripped_evaluated_boxes.sort(key=lambda x: x[aggregate_key_name])
        for key, grouped_evaluated_boxes in groupby(
            stripped_evaluated_boxes, key=lambda x: x[aggregate_key_name]
        ):
            aggregated_evaluation_results[key] = __count_evaluation_results(
                grouped_evaluated_boxes
            )

        return aggregated_evaluation_results

    raise InvalidAggregateKeyNameException(
        f"invalid aggregate key name: {aggregate_key_name}"
    )


def filter_loose_results(
    evaluated_boxes: List[ImageEvaluatedBox],
) -> List[ImageEvaluatedBox]:
    return [
        eb
        for eb in evaluated_boxes
        if (not eb.evaluated_box.loose)
        or (eb.evaluated_box.evaluation_result == EvaluationResult.CORRECT)
    ]


def filter_loose_results_strict(
    evaluated_boxes: List[ImageEvaluatedBox],
) -> List[ImageEvaluatedBox]:
    return [eb for eb in evaluated_boxes if not eb.evaluated_box.loose]


def __count_evaluation_results(evaluated_boxes):
    evaluated_boxes = sorted(evaluated_boxes, key=lambda x: x["result"])
    results_count: Dict[EvaluationResult, int] = {
        result: len(list(boxes))
        for result, boxes in groupby(evaluated_boxes, key=lambda x: x["result"])
    }

    results_images_count: Dict[EvaluationResult, int] = {
        result: len({b["image_id"] for b in boxes})
        for result, boxes in groupby(evaluated_boxes, key=lambda x: x["result"])
    }

    evaluated_boxes = sorted(
        evaluated_boxes, key=lambda x: (x["image_id"], x["result"])
    )
    images_results_pairs = [
        image_result_pair
        for image_result_pair, _ in groupby(
            evaluated_boxes, key=lambda x: (x["image_id"], x["result"])
        )
    ]
    images_results_pairs = sorted(images_results_pairs)

    image_results = {
        image_id: [r[1] for r in results]
        for image_id, results in groupby(images_results_pairs, key=lambda x: x[0])
    }

    image_gt_pred_results = {
        image_id: (
            EvaluationResult.MISSED in results
            or EvaluationResult.CORRECT in results,  # gt results
            EvaluationResult.CORRECT in results
            or EvaluationResult.DUPLICATE in results
            or EvaluationResult.WRONG in results
            or EvaluationResult.EXTRA in results,  # pred results
        )
        for image_id, results in image_results.items()
    }

    images_count = len(image_gt_pred_results.keys())
    gt_images_count = len(
        [image for image, results in image_gt_pred_results.items() if results[0]]
    )
    gt_and_pred_images_count = len(
        [
            image
            for image, results in image_gt_pred_results.items()
            if results[0] and results[1]
        ]
    )

    return AggregatedEvaluationResults(
        results_count,
        results_images_count,
        images_count,
        gt_images_count,
        gt_and_pred_images_count,
    )


def calc_evaluation_metrics_per_key(
    aggregated_evaluation_results: Dict[str, AggregatedEvaluationResults],
) -> Dict[str, EvaluationMetrics]:
    return {
        key: calc_evaluation_metrics(results)
        for key, results in aggregated_evaluation_results.items()
    }


def calc_evaluation_metrics(
    aggregated_evaluation_results: AggregatedEvaluationResults,
) -> EvaluationMetrics:
    # pylint: disable=invalid-name
    results_count = aggregated_evaluation_results.results_count

    TP = results_count.get(EvaluationResult.CORRECT, 0)  # noqa: N806
    FN = results_count.get(EvaluationResult.MISSED, 0)  # noqa: N806
    FP = (  # noqa: N806
        results_count.get(EvaluationResult.EXTRA, 0)
        + results_count.get(EvaluationResult.DUPLICATE, 0)
        + results_count.get(EvaluationResult.WRONG, 0)
    )

    gt_and_pred_images_count = aggregated_evaluation_results.gt_and_pred_images_count
    images_count = aggregated_evaluation_results.images_count

    return EvaluationMetrics(
        TP=TP,
        FN=FN,
        FP=FP,
        GT_Count=TP + FN,
        Precision=TP / (TP + FP) if (TP + FP) > 0 else None,
        Recall=TP / (TP + FN) if (TP + FN) > 0 else None,
        AK=TP / (TP + FN + FP) if (TP + FN + FP) > 0 else None,
        OB=gt_and_pred_images_count / images_count if images_count > 0 else None,
        images_count=images_count,
    )


def calc_global_weighted_value(
    value_fn: Callable,
    class_evaluation_metrics: Dict[str, EvaluationMetrics],
    weight_fn: Callable,
    ak_precision=None,
) -> float:
    base = sum(
        (
            weight_fn(m) or 0
            for m in class_evaluation_metrics.values()
            if not math.isnan(weight_fn(m) or 0)
        )
    )
    if base == 0:
        return 0

    if ak_precision:
        numerator = sum(
            (
                (round(value_fn(m), ak_precision) or 0) * (weight_fn(m) or 0)
                for m in class_evaluation_metrics.values()
                if not math.isnan(
                    (round(value_fn(m), ak_precision) or 0) * (weight_fn(m) or 0)
                )
            )
        )
    else:
        numerator = sum(
            (
                (value_fn(m) or 0) * (weight_fn(m) or 0)
                for m in class_evaluation_metrics.values()
                if not math.isnan((value_fn(m) or 0) * (weight_fn(m) or 0))
            )
        )

    return numerator / base
