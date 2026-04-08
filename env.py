"""
env.py — OpenEnv-compatible reinforcement-learning environment for code review.

Implements the canonical RL loop:
    obs  = env.reset(task_id)
    while not env.is_done():
        obs    = env.observe()
        reward = env.step(action)
    final  = env.episode_reward()

The environment is stateful: one instance tracks a single episode at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graders import GradeResult, compute_episode_reward, grade_action
from tasks import TASK_REGISTRY, Task, get_task


# ---------------------------------------------------------------------------
# Observation / Info helpers
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """What the agent sees at each step."""
    task_name: str
    code_snippet: str
    instructions: str
    step: int
    total_steps: int

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "code_snippet": self.code_snippet,
            "instructions": self.instructions,
            "step": self.step,
            "total_steps": self.total_steps,
        }

    def to_prompt(self) -> str:
        """Render a human-readable prompt for the LLM agent."""
        return (
            f"## Code Review Task: {self.task_name}\n"
            f"**Step {self.step}/{self.total_steps}**\n\n"
            f"### Code Under Review\n```python\n{self.code_snippet}\n```\n\n"
            f"### Instructions\n{self.instructions}\n\n"
            "### Required Response Format\n"
            "Respond with a JSON object containing:\n"
            '- `"bug_description"`: A clear description of the identified bug(s).\n'
            '- `"severity"`: One of "low", "medium", "high", "critical".\n'
            '- `"suggested_fix"`: A concrete suggestion for how to fix the issue.\n\n'
            "Respond ONLY with the JSON object — no extra text."
        )


@dataclass
class StepInfo:
    """Returned to the caller after each step."""
    reward: float
    done: bool
    observation: Optional[Observation]
    grade_result: GradeResult
    cumulative_reward: float

    def to_dict(self) -> dict:
        return {
            "reward": self.reward,
            "done": self.done,
            "observation": self.observation.to_dict() if self.observation else None,
            "grade_reason": self.grade_result.reason,
            "matched_bugs": self.grade_result.matched_bugs,
            "cumulative_reward": self.cumulative_reward,
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CodeReviewEnv:
    """
    OpenEnv-compatible code-review environment.

    Lifecycle:
        1. ``reset(task_id)``  — initialise a new episode, receive first observation.
        2. ``step(action)``    — submit an action, receive reward + next observation.
        3. ``is_done()``       — check whether the episode has ended.
        4. ``episode_reward()``— aggregate reward for the episode.
    """

    def __init__(self) -> None:
        self._task: Optional[Task] = None
        self._current_step: int = 0
        self._done: bool = True
        self._results: List[GradeResult] = []
        self._actions: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        """
        Start a new episode for the given task.

        Args:
            task_id: Identifier of the task (e.g. ``"easy_offbyone"``).

        Returns:
            The first ``Observation`` the agent should act on.
        """
        self._task = get_task(task_id)
        self._current_step = 0
        self._done = False
        self._results = []
        self._actions = []
        return self._make_observation()

    def observe(self) -> Observation:
        """Return the current observation without advancing state."""
        self._assert_running()
        return self._make_observation()

    def step(self, action: dict) -> StepInfo:
        """
        Submit an action and advance the environment by one step.

        Args:
            action: Dict with keys ``bug_description``, ``severity``,
                    ``suggested_fix``.

        Returns:
            ``StepInfo`` containing the reward, whether the episode is done,
            the next observation (if any), and grading details.
        """
        self._assert_running()

        task_step = self._task.steps[self._current_step]
        result = grade_action(
            action=action,
            step=task_step,
            current_step=self._current_step,
            max_steps=self._task.max_steps,
        )

        self._results.append(result)
        self._actions.append(action)
        self._current_step += 1

        if self._current_step >= len(self._task.steps):
            self._done = True

        next_obs = None if self._done else self._make_observation()

        return StepInfo(
            reward=result.reward,
            done=self._done,
            observation=next_obs,
            grade_result=result,
            cumulative_reward=self.episode_reward(),
        )

    def is_done(self) -> bool:
        """Return ``True`` if the current episode has ended."""
        return self._done

    def episode_reward(self) -> float:
        """Return the mean reward across all completed steps."""
        return compute_episode_reward(self._results)

    def episode_summary(self) -> dict:
        """Return a JSON-serialisable summary of the completed episode."""
        return {
            "task_id": self._task.task_id if self._task else None,
            "task_name": self._task.task_name if self._task else None,
            "total_steps": self._current_step,
            "episode_reward": self.episode_reward(),
            "step_results": [
                {
                    "step": i + 1,
                    "reward": r.reward,
                    "bug_detection": r.bug_detection_score,
                    "severity": r.severity_score,
                    "fix_relevance": r.fix_score,
                    "penalty": r.step_penalty_applied,
                    "reason": r.reason,
                    "matched_bugs": r.matched_bugs,
                }
                for i, r in enumerate(self._results)
            ],
        }

    @staticmethod
    def list_tasks() -> list[dict]:
        """List all available tasks."""
        from tasks import list_tasks
        return list_tasks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        """Build an ``Observation`` for the current step."""
        assert self._task is not None
        step_def = self._task.steps[self._current_step]
        return Observation(
            task_name=self._task.task_name,
            code_snippet=self._task.code_snippet,
            instructions=step_def.instruction,
            step=self._current_step + 1,
            total_steps=len(self._task.steps),
        )

    def _assert_running(self) -> None:
        """Raise if no episode is active."""
        if self._done or self._task is None:
            raise RuntimeError(
                "No active episode. Call env.reset(task_id) first."
            )
