"""
graders.py — Deterministic grading engine for the Code Review RL Environment.

Grading components:
    1. Bug detection accuracy   — keyword matching against ground-truth bug descriptions.
    2. Severity correctness     — exact match on severity level.
    3. Fix relevance            — keyword matching on the suggested fix.
    4. Step penalty             — small deduction for each step taken (encourages efficiency).

All scores are normalised to [0, 1].
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from tasks import GroundTruthBug, Severity, TaskStep


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STEP_PENALTY: float = 0.02          # Deducted per step (max_steps = 1 → no penalty)
BUG_DETECTION_WEIGHT: float = 0.50  # Weight of bug-detection component
SEVERITY_WEIGHT: float = 0.20      # Weight of severity component
FIX_WEIGHT: float = 0.30           # Weight of fix-relevance component


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    """Result of grading a single agent action."""
    reward: float                    # Combined score in [0, 1]
    bug_detection_score: float       # Sub-score for detection
    severity_score: float            # Sub-score for severity
    fix_score: float                 # Sub-score for fix relevance
    step_penalty_applied: float      # Penalty deducted
    reason: str                      # Human-readable explanation
    matched_bugs: List[str]          # Descriptions of matched ground-truth bugs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace for fuzzy keyword matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _keyword_match(text: str, keywords: List[str]) -> bool:
    """Return True if *any* keyword is found in text (case-insensitive)."""
    text_norm = _normalise(text)
    return any(kw.lower() in text_norm for kw in keywords)


def _severity_distance(predicted: str, expected: Severity) -> float:
    """
    Return a similarity score between predicted and expected severity.
        Exact match  → 1.0
        One level off → 0.5
        Two levels   → 0.2
        Else         → 0.0
    """
    levels = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    try:
        pred_idx = levels.index(Severity(predicted.lower()))
    except (ValueError, KeyError):
        return 0.0
    exp_idx = levels.index(expected)
    diff = abs(pred_idx - exp_idx)
    return {0: 1.0, 1: 0.5, 2: 0.2}.get(diff, 0.0)


# ---------------------------------------------------------------------------
# Core grading
# ---------------------------------------------------------------------------

def grade_action(
    action: dict,
    step: TaskStep,
    current_step: int,
    max_steps: int,
) -> GradeResult:
    """
    Grade a single agent action against the ground-truth for a task step.

    Args:
        action:       Parsed agent response with keys
                      ``bug_description``, ``severity``, ``suggested_fix``.
        step:         The ``TaskStep`` the agent is responding to.
        current_step: 0-indexed step number.
        max_steps:    Total number of steps for the task.

    Returns:
        ``GradeResult`` with a composite reward in [0, 1].
    """
    bug_desc: str = str(action.get("bug_description", ""))
    predicted_severity: str = str(action.get("severity", ""))
    suggested_fix: str = str(action.get("suggested_fix", ""))

    # Combine all agent text for broader matching
    agent_text = f"{bug_desc} {suggested_fix}"

    # --- 1. Bug detection (weighted across ground-truth bugs) ----------------
    total_weight = sum(gt.weight for gt in step.ground_truth_bugs)
    detection_score = 0.0
    severity_score = 0.0
    fix_score = 0.0
    matched: List[str] = []

    for gt_bug in step.ground_truth_bugs:
        detected = _keyword_match(agent_text, gt_bug.bug_keywords)
        if detected:
            matched.append(gt_bug.description)
            detection_score += gt_bug.weight / total_weight

            # Severity scored only if bug was detected
            severity_score += (
                _severity_distance(predicted_severity, gt_bug.expected_severity)
                * gt_bug.weight / total_weight
            )

            # Fix relevance scored only if bug was detected
            if _keyword_match(suggested_fix, gt_bug.fix_keywords):
                fix_score += gt_bug.weight / total_weight

    # Clamp sub-scores to [0, 1]
    detection_score = min(detection_score, 1.0)
    severity_score = min(severity_score, 1.0)
    fix_score = min(fix_score, 1.0)

    # --- 2. Step penalty -----------------------------------------------------
    penalty = STEP_PENALTY * current_step

    # --- 3. Composite reward -------------------------------------------------
    raw = (
        BUG_DETECTION_WEIGHT * detection_score
        + SEVERITY_WEIGHT * severity_score
        + FIX_WEIGHT * fix_score
    )
    reward = max(0.0, min(1.0, raw - penalty))

    # Validator requires scores strictly in (0, 1) — clamp to open interval
    EPSILON = 0.01
    if reward <= 0.0:
        reward = EPSILON
    elif reward >= 1.0:
        reward = 1.0 - EPSILON

    # --- 4. Reason string ----------------------------------------------------
    reasons = []
    if detection_score > 0:
        reasons.append(f"Detected {len(matched)}/{len(step.ground_truth_bugs)} bugs")
    else:
        reasons.append("No known bugs detected")
    if severity_score > 0:
        reasons.append(f"severity score {severity_score:.2f}")
    if fix_score > 0:
        reasons.append(f"fix relevance {fix_score:.2f}")
    if penalty > 0:
        reasons.append(f"step penalty -{penalty:.2f}")
    reason = "; ".join(reasons) + f" → reward={reward:.4f}"

    return GradeResult(
        reward=round(reward, 4),
        bug_detection_score=round(detection_score, 4),
        severity_score=round(severity_score, 4),
        fix_score=round(fix_score, 4),
        step_penalty_applied=round(penalty, 4),
        reason=reason,
        matched_bugs=matched,
    )


def grade_episode(
    actions: List[dict],
    steps: List[TaskStep],
    max_steps: int,
) -> List[GradeResult]:
    """
    Grade an entire episode (all steps).

    Returns one ``GradeResult`` per step.
    """
    results: List[GradeResult] = []
    for idx, (action, step) in enumerate(zip(actions, steps)):
        results.append(grade_action(action, step, idx, max_steps))
    return results


def compute_episode_reward(results: List[GradeResult]) -> float:
    """Return the mean reward across all steps of an episode."""
    if not results:
        return 0.01  # Validator requires strictly > 0
    mean = sum(r.reward for r in results) / len(results)
    # Clamp to open interval (0, 1)
    mean = max(0.01, min(0.99, mean))
    return round(mean, 4)

