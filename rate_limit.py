# ABOUTME: Rate limiting logic and scores.json management for hackathon submissions.
# ABOUTME: Handles hourly/total limits, deadline checks, submission recording, and leaderboard generation.
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class RateLimitCheck:
    """Result of a rate limit check."""

    allowed: bool
    reason: str
    remaining_hourly: int
    remaining_total: int


@dataclass(frozen=True)
class SubmissionRecord:
    """A single submission entry for scores.json."""

    timestamp: str
    sha: str
    map_at_05: float | None
    predictions_path: str
    status: str  # "scored" | "rate_limited" | "error" | "past_deadline"


def check_rate_limit(team_id: str, scores_data: dict, now: datetime) -> RateLimitCheck:
    """Check whether a team is allowed to submit based on hourly and total limits.

    Counts only submissions with status "scored" toward limits.
    """
    per_hour = scores_data["limits"]["per_hour"]
    total_limit = scores_data["limits"]["total"]

    team = scores_data.get("teams", {}).get(team_id)
    if team is None:
        return RateLimitCheck(
            allowed=True,
            reason="First submission",
            remaining_hourly=per_hour - 1,
            remaining_total=total_limit - 1,
        )

    submissions = team.get("submissions", [])
    hour_ago = now - timedelta(hours=1)

    hourly_count = sum(
        1
        for s in submissions
        if datetime.fromisoformat(s["timestamp"]) > hour_ago
    )
    total_count = len(submissions)

    remaining_hourly = max(0, per_hour - hourly_count)
    remaining_total = max(0, total_limit - total_count)

    if remaining_total == 0:
        return RateLimitCheck(
            allowed=False,
            reason=f"Total submission limit ({total_limit}) reached",
            remaining_hourly=remaining_hourly,
            remaining_total=0,
        )

    if remaining_hourly == 0:
        return RateLimitCheck(
            allowed=False,
            reason=f"Hourly submission limit ({per_hour}) reached",
            remaining_hourly=0,
            remaining_total=remaining_total,
        )

    return RateLimitCheck(
        allowed=True,
        reason="OK",
        remaining_hourly=remaining_hourly - 1,
        remaining_total=remaining_total - 1,
    )


def check_deadline(deadline_iso: str, now: datetime) -> bool:
    """Return True if the current time is strictly before the deadline."""
    deadline = datetime.fromisoformat(deadline_iso)
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now < deadline


def record_submission(
    team_id: str, scores_data: dict, record: SubmissionRecord
) -> dict:
    """Record a submission in scores_data, updating best_map if appropriate.

    Returns the updated scores_data (mutated in place).
    """
    teams = scores_data.setdefault("teams", {})
    if team_id not in teams:
        teams[team_id] = {
            "submissions": [],
            "best_map": 0.0,
            "total_scored": 0,
            "total_submissions": 0,
        }

    team = teams[team_id]
    team["submissions"].append(asdict(record))
    team["total_submissions"] += 1

    if record.status == "scored" and record.map_at_05 is not None:
        team["total_scored"] += 1
        if record.map_at_05 > team["best_map"]:
            team["best_map"] = record.map_at_05

    return scores_data


def scores_to_leaderboard(scores_data: dict) -> list[dict]:
    """Convert scores_data to a leaderboard sorted by best_map descending."""
    teams = scores_data.get("teams", {})
    entries = []
    for team_id, team in teams.items():
        submissions = team.get("submissions", [])
        last_sub = submissions[-1]["timestamp"] if submissions else None
        entries.append({
            "team_id": team_id,
            "best_map": team["best_map"],
            "total_submissions": team["total_submissions"],
            "last_submission": last_sub,
        })
    entries.sort(key=lambda e: e["best_map"], reverse=True)
    return entries


def load_scores(path: Path) -> dict:
    """Load scores.json from disk, returning empty structure if file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return {
            "deadline": "",
            "limits": {"per_hour": 5, "total": 30},
            "teams": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_scores(scores_data: dict, path: Path) -> None:
    """Write scores_data to disk as pretty-printed JSON."""
    Path(path).write_text(json.dumps(scores_data, indent=2) + "\n", encoding="utf-8")
