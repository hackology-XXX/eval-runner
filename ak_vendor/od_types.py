# ABOUTME: Object-detection metric value types (AK weighting + per-class metrics).
# ABOUTME: Vendored subset of assecobs_quality_metrics @ 7fa3f25 (2026-06-03) — pandas/face/SLA stripped.
from typing import Dict, Optional
from enum import IntEnum
from dataclasses import dataclass

from .types import EvaluationResult


@dataclass
class EvaluationMetrics:  # pylint: disable=too-many-instance-attributes,invalid-name
    TP: int
    FN: int
    FP: int
    GT_Count: int
    Precision: Optional[float]
    Recall: Optional[float]
    AK: Optional[float]
    OB: Optional[float]
    images_count: int


@dataclass
class AggregatedEvaluationResults:
    results_count: Dict[EvaluationResult, int]
    results_images_count: Dict[EvaluationResult, int]
    images_count: int
    gt_images_count: int
    gt_and_pred_images_count: int


class AvgWeightMode(IntEnum):
    GROUND_TRUTH = 0
    GROUND_TRUTH_AND_PREDICTIONS = 1
    TP_FN_FP = 2
    EQUAL = 3
    IMAGES_COUNT = 4
