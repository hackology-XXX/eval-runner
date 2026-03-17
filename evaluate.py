# ABOUTME: Orchestrator script for evaluating hackathon submissions via GitHub Actions.
# ABOUTME: Reads env vars, fetches predictions, validates, scores, records results, sets commit status.
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# These modules are copied into eval-runner at deploy time
from score import ScoreResult, compute_map
from rate_limit import (
    SubmissionRecord,
    check_deadline,
    check_rate_limit,
    load_scores,
    record_submission,
    save_scores,
    scores_to_leaderboard,
)
from validate_predictions import validate_predictions


SCORES_PATH = Path("scores.json")
GT_ENCRYPTED_PATH = Path("ground_truth.json.gpg")
GT_PASSPHRASE_ENV = "GT_PASSPHRASE"


def _env(name: str) -> str:
    """Read required environment variable or exit."""
    value = os.environ.get(name)
    if not value:
        print(f"ERROR: Missing required env var: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def _set_commit_status(team_repo: str, sha: str, state: str, description: str) -> None:
    """Set commit status on the team's repo via gh CLI."""
    subprocess.run(
        [
            "gh", "api",
            f"/repos/{team_repo}/statuses/{sha}",
            "-X", "POST",
            "-f", f"state={state}",
            "-f", f"description={description}",
            "-f", "context=hackology/eval",
        ],
        check=False,
        capture_output=True,
    )


def _fetch_predictions(team_repo: str, sha: str, predictions_path: str) -> list[dict]:
    """Fetch predictions file from team repo at specific commit."""
    result = subprocess.run(
        [
            "gh", "api",
            f"/repos/{team_repo}/contents/{predictions_path}",
            "-H", "Accept: application/vnd.github.raw+json",
            "--jq", ".",
            "-X", "GET",
            "-H", f"X-GitHub-Api-Version: 2022-11-28",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GH_REF": sha},
    )
    return json.loads(result.stdout)


def _decrypt_ground_truth(passphrase: str) -> dict:
    """Decrypt GPG-encrypted ground truth file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    subprocess.run(
        [
            "gpg", "--batch", "--yes", "--passphrase", passphrase,
            "--output", tmp_path, "--decrypt", str(GT_ENCRYPTED_PATH),
        ],
        check=True,
        capture_output=True,
    )
    gt = json.loads(Path(tmp_path).read_text(encoding="utf-8"))
    Path(tmp_path).unlink()
    return gt


def main() -> None:
    """Main evaluation flow triggered by repository_dispatch."""
    team_id = _env("TEAM_ID")
    team_repo = _env("TEAM_REPO")
    sha = _env("SHA")
    predictions_path = _env("PREDICTIONS_PATH")
    gt_passphrase = _env(GT_PASSPHRASE_ENV)

    now = datetime.now(timezone.utc)
    scores_data = load_scores(SCORES_PATH)

    # Check deadline
    if scores_data.get("deadline") and not check_deadline(scores_data["deadline"], now):
        record = SubmissionRecord(
            timestamp=now.isoformat(),
            sha=sha,
            map_at_05=None,
            predictions_path=predictions_path,
            status="past_deadline",
        )
        record_submission(team_id, scores_data, record)
        save_scores(scores_data, SCORES_PATH)
        _set_commit_status(team_repo, sha, "failure", "Submission deadline has passed")
        print("Submission rejected: past deadline")
        sys.exit(0)

    # Check rate limit
    limit_check = check_rate_limit(team_id, scores_data, now)
    if not limit_check.allowed:
        record = SubmissionRecord(
            timestamp=now.isoformat(),
            sha=sha,
            map_at_05=None,
            predictions_path=predictions_path,
            status="rate_limited",
        )
        record_submission(team_id, scores_data, record)
        save_scores(scores_data, SCORES_PATH)
        _set_commit_status(team_repo, sha, "failure", f"Rate limited: {limit_check.reason}")
        print(f"Submission rejected: {limit_check.reason}")
        sys.exit(0)

    # Fetch and validate predictions
    try:
        predictions = _fetch_predictions(team_repo, sha, predictions_path)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        record = SubmissionRecord(
            timestamp=now.isoformat(),
            sha=sha,
            map_at_05=None,
            predictions_path=predictions_path,
            status="error",
        )
        record_submission(team_id, scores_data, record)
        save_scores(scores_data, SCORES_PATH)
        _set_commit_status(team_repo, sha, "error", f"Failed to fetch predictions: {e}")
        print(f"ERROR: Failed to fetch predictions: {e}", file=sys.stderr)
        sys.exit(1)

    # Decrypt ground truth and extract valid IDs for validation
    gt = _decrypt_ground_truth(gt_passphrase)
    valid_image_ids = {img["id"] for img in gt["images"]}
    valid_category_ids = {cat["id"] for cat in gt["categories"]}

    validation = validate_predictions(predictions, valid_image_ids, valid_category_ids)
    if not validation.is_valid:
        record = SubmissionRecord(
            timestamp=now.isoformat(),
            sha=sha,
            map_at_05=None,
            predictions_path=predictions_path,
            status="error",
        )
        record_submission(team_id, scores_data, record)
        save_scores(scores_data, SCORES_PATH)
        errors_str = "; ".join(validation.errors[:3])
        _set_commit_status(team_repo, sha, "failure", f"Validation failed: {errors_str}")
        print(f"Validation failed: {errors_str}", file=sys.stderr)
        sys.exit(1)

    # Score predictions
    score_result = compute_map(predictions, gt)

    record = SubmissionRecord(
        timestamp=now.isoformat(),
        sha=sha,
        map_at_05=score_result.map_at_05,
        predictions_path=predictions_path,
        status="scored",
    )
    record_submission(team_id, scores_data, record)
    save_scores(scores_data, SCORES_PATH)

    desc = f"mAP@0.5: {score_result.map_at_05:.4f}"
    _set_commit_status(team_repo, sha, "success", desc)
    print(f"Scored: {desc} (mAP@0.5:0.95: {score_result.map_at_05_095:.4f})")
    print(f"Predictions: {score_result.n_predictions}, GT: {score_result.n_ground_truth}")


if __name__ == "__main__":
    main()
