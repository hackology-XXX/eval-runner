# ABOUTME: Offline validation of predictions.json submissions for hackathon scoring pipeline.
# ABOUTME: Checks structure, bbox format, score range, image/category IDs, and file size.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

REQUIRED_KEYS = {"image_id", "category_id", "bbox", "score"}


@dataclass(frozen=True)
class ValidationResult:
    """Result of predictions validation."""

    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_predictions(
    predictions: list[dict],
    valid_image_ids: set[int],
    valid_category_ids: set[int],
) -> ValidationResult:
    """Validate a list of prediction dicts against known image and category IDs.

    Returns a ValidationResult with all errors found (not just the first).
    """
    errors: list[str] = []

    if not isinstance(predictions, list):
        return ValidationResult(errors=["Predictions must be a list"])

    for i, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            errors.append(f"Entry {i}: must be a dict, got {type(pred).__name__}")
            continue

        # Check required keys
        missing = REQUIRED_KEYS - pred.keys()
        if missing:
            errors.append(f"Entry {i}: missing keys: {', '.join(sorted(missing))}")
            continue

        # Validate bbox
        bbox = pred["bbox"]
        if not isinstance(bbox, (list, tuple)):
            errors.append(f"Entry {i}: bbox must be a list, got {type(bbox).__name__}")
        elif len(bbox) != 4:
            errors.append(f"Entry {i}: bbox must have 4 elements, got {len(bbox)}")
        else:
            try:
                values = [float(v) for v in bbox]
            except (TypeError, ValueError):
                errors.append(f"Entry {i}: bbox contains non-numeric values")
            else:
                if values[2] <= 0 or values[3] <= 0:
                    errors.append(f"Entry {i}: bbox width and height must be > 0, got {bbox}")

        # Validate score
        score = pred["score"]
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            errors.append(f"Entry {i}: score must be a number, got {type(score).__name__}")
        else:
            if score_f <= 0 or score_f > 1:
                errors.append(f"Entry {i}: score must be in (0, 1], got {score_f}")

        # Validate image_id
        if pred["image_id"] not in valid_image_ids:
            errors.append(f"Entry {i}: unknown image_id {pred['image_id']}")

        # Validate category_id
        if pred["category_id"] not in valid_category_ids:
            errors.append(f"Entry {i}: unknown category_id {pred['category_id']}")

    return ValidationResult(errors=errors)


def validate_predictions_file(
    path: Path,
    valid_image_ids: set[int],
    valid_category_ids: set[int],
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> ValidationResult:
    """Validate a predictions.json file from disk.

    Checks file existence, size limit, JSON parsing, then delegates to validate_predictions.
    """
    path = Path(path)

    if not path.exists():
        return ValidationResult(errors=[f"File not found: {path}"])

    file_size = path.stat().st_size
    if file_size > max_size_bytes:
        return ValidationResult(
            errors=[f"File size {file_size} bytes exceeds limit of {max_size_bytes} bytes"]
        )

    try:
        text = path.read_text(encoding="utf-8")
        predictions = json.loads(text)
    except json.JSONDecodeError as e:
        return ValidationResult(errors=[f"Invalid JSON: {e}"])

    return validate_predictions(predictions, valid_image_ids, valid_category_ids)
