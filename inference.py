"""
inference.py — Main entry point for the Code Review RL agent.

Runs the full RL loop:
    1. Initialise the environment for each task.
    2. Send observations to the LLM via the OpenAI-compatible API.
    3. Parse structured JSON actions from the model response.
    4. Step the environment and collect rewards.
    5. Print output in the required [START] / [STEP] / [END] format.

Environment variables:
    API_BASE_URL  — Base URL for the OpenAI-compatible API (default: https://api.openai.com/v1)
    MODEL_NAME    — Model identifier (default: gpt-4o)
    HF_TOKEN      — Hugging Face token (used for API access)
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from typing import Optional

from env import CodeReviewEnv
from tasks import TASK_REGISTRY


# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# Flag to track whether we can use the LLM
_LLM_AVAILABLE = True

if not HF_TOKEN:
    print("WARNING: HF_TOKEN environment variable not set. Using fallback responses.", file=sys.stderr)
    _LLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# OpenAI client (lazy initialisation)
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """Lazily initialise the OpenAI client."""
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            _client = OpenAI(
                base_url=API_BASE_URL,
                api_key=HF_TOKEN,
            )
        except Exception as e:
            print(f"WARNING: Failed to initialise OpenAI client: {e}", file=sys.stderr)
            return None
    return _client


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


def _extract_json(text: str) -> Optional[dict]:
    """
    Attempt to extract a JSON object from the model's response.

    Strategy:
        1. Look for a fenced ```json ... ``` code block.
        2. Look for the first { ... } substring.
        3. Return None on failure.
    """
    # Strategy 1: fenced code block
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 2: first { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _safe_parse_action(text: str) -> dict:
    """
    Parse the model output into a valid action dict.
    Returns a fallback with empty strings if parsing fails entirely.

    Handles cases where the model returns:
        - A single JSON object  → used directly
        - A JSON array of objects → merged into one combined action
        - Raw text → used as-is for partial grading
    """
    result = _extract_json(text)

    if result is None:
        # Use the raw text as the bug description so partial grading can still work
        return {
            "bug_description": text,
            "severity": "medium",
            "suggested_fix": text,
        }

    # If the model returned a list of bug objects, merge them
    if isinstance(result, list):
        descriptions = []
        severities = []
        fixes = []
        for item in result:
            if isinstance(item, dict):
                descriptions.append(str(item.get("bug_description", "")))
                severities.append(str(item.get("severity", "")))
                fixes.append(str(item.get("suggested_fix", "")))
        # Pick the highest severity from the list
        severity_order = ["critical", "high", "medium", "low"]
        best_severity = "medium"
        for s in severity_order:
            if s in severities:
                best_severity = s
                break
        return {
            "bug_description": " | ".join(filter(None, descriptions)),
            "severity": best_severity,
            "suggested_fix": " | ".join(filter(None, fixes)),
        }

    return {
        "bug_description": str(result.get("bug_description", "")),
        "severity": str(result.get("severity", "medium")),
        "suggested_fix": str(result.get("suggested_fix", "")),
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert code reviewer. You review code for bugs, security "
    "vulnerabilities, and design issues. Always respond with a single JSON "
    "object containing exactly three keys: "
    '"bug_description", "severity" (one of: low, medium, high, critical), '
    'and "suggested_fix". Do NOT include any extra text outside the JSON.'
)


def _fallback_response(prompt: str) -> str:
    """Generate a reasonable fallback response when LLM is unavailable."""
    return json.dumps({
        "bug_description": "Potential bug detected in the code logic",
        "severity": "medium",
        "suggested_fix": "Review the code for logical errors and edge cases",
    })


def call_llm(prompt: str) -> str:
    """Send a prompt to the LLM and return the raw response text."""
    if not _LLM_AVAILABLE:
        return _fallback_response(prompt)

    client = _get_client()
    if client is None:
        return _fallback_response(prompt)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"WARNING: LLM call failed: {e}", file=sys.stderr)
        return _fallback_response(prompt)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> None:
    """
    Execute the full inference loop across all tasks.

    Output format (strict):
        [START]
        [STEP] {"task_id": ..., "step": ..., "reward": ..., ...}
        [STEP] ...
        [END]
    """
    env = CodeReviewEnv()
    task_ids = list(TASK_REGISTRY.keys())

    print("[START]")

    for task_id in task_ids:
        try:
            obs = env.reset(task_id)

            while not env.is_done():
                prompt = obs.to_prompt()
                raw_response = call_llm(prompt)
                action = _safe_parse_action(raw_response)
                step_info = env.step(action)

                step_output = {
                    "task_id": task_id,
                    "step": step_info.grade_result and (env._current_step),
                    "reward": step_info.reward,
                    "cumulative_reward": step_info.cumulative_reward,
                    "done": step_info.done,
                    "reason": step_info.grade_result.reason,
                    "matched_bugs": step_info.grade_result.matched_bugs,
                    "action": action,
                }
                print(f"[STEP] {json.dumps(step_output)}")

                if not step_info.done and step_info.observation:
                    obs = step_info.observation

            # Episode summary
            summary = env.episode_summary()
            print(f"[STEP] {json.dumps({'episode_summary': summary})}")

        except Exception as e:
            print(f"[STEP] {json.dumps({'task_id': task_id, 'error': str(e)})}", file=sys.stdout)
            print(f"WARNING: Task {task_id} failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print("[END]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_inference()
    except Exception as e:
        # Last-resort error handling — print [START]/[END] even on crash
        print(f"FATAL: inference.py crashed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("[START]")
        print(f'[STEP] {json.dumps({"error": str(e)})}')
        print("[END]")
        sys.exit(1)
