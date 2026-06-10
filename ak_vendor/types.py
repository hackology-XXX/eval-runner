# ABOUTME: Core bounding-box / evaluation dataclasses for the AK metric.
# ABOUTME: Vendored from assecobs_quality_metrics @ 7fa3f25 (2026-06-03) — DO NOT EDIT LOGIC.
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Union


@dataclass
class BoundingBox:  # pylint: disable=too-many-instance-attributes
    left: Union[int, float]
    top: Union[int, float]
    right: Union[int, float]
    bottom: Union[int, float]
    label: str
    confidence: Optional[float]
    loose: bool
    face: bool = False

    def __post_init__(self):
        failed_validations = []
        if self.left >= self.right:
            failed_validations.append("xmin greater or equal to xmax")
        if self.top >= self.bottom:
            failed_validations.append("ymin greater or equal to ymax")
        if len(failed_validations) > 0:
            raise ValueError(failed_validations)

    def __hash__(self) -> int:
        return hash(
            (
                self.left,
                self.top,
                self.right,
                self.bottom,
                self.label,
                self.confidence,
                self.loose,
                self.face,
            )
        )


@dataclass
class PredictionBoundingBox(BoundingBox):
    product: Optional[str] = ""

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.product))


@dataclass
class GroundTruthPredictionsPair:
    ground_truth: List[BoundingBox]
    predictions: List[PredictionBoundingBox]
    image_id: str

    def __hash__(self) -> int:
        return hash((tuple(self.ground_truth), tuple(self.predictions), self.image_id))


class EvaluationResult(IntEnum):
    CORRECT = 0
    MISSED = 1
    EXTRA = 2
    DUPLICATE = 3
    WRONG = 4

    def __str__(self):
        return str(self.name.lower())


@dataclass
class EvaluatedBox:
    bounding_box: BoundingBox
    evaluation_result: EvaluationResult
    ground_truth_label: Optional[str]
    loose: bool


@dataclass
class ImageEvaluatedBox:
    evaluated_box: EvaluatedBox
    image_id: str


@dataclass
class EvaluatedBoxWithIndex(EvaluatedBox):
    """Rozszerza EvaluatedBox o indeks sparowanego ground-truth bboxa.

    matched_gt_index to 0-based indeks do per-image listy
    GroundTruthPredictionsPair.ground_truth. Semantyka per EvaluationResult:

    - CORRECT: indeks GT bboxa z którym predykcja została sparowana po IoU.
    - WRONG: indeks GT bboxa nakrywanego przez predykcję (IoU >= threshold),
      mimo że klasa predykcji nie zgadza się z klasą GT.
    - DUPLICATE: indeks GT bboxa który predykcja nakryła — uwaga: inna predykcja
      tej samej klasy już zajęła ten slot GT, więc konsumenci grupujący po
      matched_gt_index muszą filtrować po evaluation_result aby uniknąć
      zawyżonych liczników per-GT.
    - EXTRA: None — żaden GT bbox nie pokrył się z IoU >= threshold.
    - MISSED: indeks samego GT (EvaluatedBoxWithIndex reprezentuje tutaj
      sam GT, nie predykcję). Pozwala konsumentom skorelować MISSED entry
      z konkretnym GT slotem — w szczególności rozróżnić MISSED-after-WRONG
      (ten sam indeks pojawia się w WRONG entry) od MISSED-truly-unseen
      (indeks unikalny). Dual-emission (WRONG + MISSED dla tego samego GT
      gdy pred ma złą klasę) jest celowe — pozostawia semantykę AK/OB
      identyczną z wcześniejszym evaluate_boxes.
    """

    matched_gt_index: Optional[int] = None


@dataclass
class ImageEvaluatedBoxWithIndex(ImageEvaluatedBox):
    """ImageEvaluatedBox z evaluated_box typu EvaluatedBoxWithIndex.

    Dziedziczy strukturę z ImageEvaluatedBox — runtime accept dowolnego
    EvaluatedBoxWithIndex w polu evaluated_box (subklasa EvaluatedBox).
    """


@dataclass
class GTBoundingBoxError(BoundingBox):
    has_prediction: bool = False
    has_match: bool = False
