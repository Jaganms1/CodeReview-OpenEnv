"""
server.py — FastAPI server wrapping the Code Review RL Environment.

Exposes the OpenEnv-compatible REST API:
    POST /reset         — Start a new episode
    POST /step          — Submit an action, receive reward + next observation
    GET  /observe       — Get current observation
    GET  /health        — Health check
    GET  /list_tasks    — List available tasks
    GET  /episode       — Get episode summary
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env import CodeReviewEnv
from tasks import TASK_REGISTRY

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Code Review Environment",
    description="OpenEnv-compatible RL environment for LLM code review agents",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global environment instance
# ---------------------------------------------------------------------------
env = CodeReviewEnv()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = Field(
        default="easy_offbyone",
        description="Task ID to start. Options: easy_offbyone, medium_auth, hard_pr_review",
    )


class ActionRequest(BaseModel):
    bug_description: str = Field(default="", description="Description of the identified bug")
    severity: str = Field(default="medium", description="Severity: low, medium, high, critical")
    suggested_fix: str = Field(default="", description="Suggested fix for the bug")


class ObservationResponse(BaseModel):
    task_name: str
    code_snippet: str
    instructions: str
    step: int
    total_steps: int


class StepResponse(BaseModel):
    reward: float
    done: bool
    observation: Optional[ObservationResponse] = None
    reason: str
    matched_bugs: List[str]
    cumulative_reward: float


class ResetResponse(BaseModel):
    observation: ObservationResponse
    task_id: str
    task_name: str
    total_steps: int


class EpisodeSummaryResponse(BaseModel):
    task_id: Optional[str]
    task_name: Optional[str]
    total_steps: int
    episode_reward: float
    step_results: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    environment: str
    available_tasks: List[str]


class TaskInfo(BaseModel):
    task_id: str
    task_name: str
    difficulty: str
    num_steps: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        environment="code-review-env",
        available_tasks=list(TASK_REGISTRY.keys()),
    )


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint — same as health check."""
    return await health_check()


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and start a new episode.

    Accepts a task_id in the request body.
    Returns the first observation.
    """
    try:
        obs = env.reset(request.task_id)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"Environment reset with task: {request.task_id}")

    return ResetResponse(
        observation=ObservationResponse(
            task_name=obs.task_name,
            code_snippet=obs.code_snippet,
            instructions=obs.instructions,
            step=obs.step,
            total_steps=obs.total_steps,
        ),
        task_id=request.task_id,
        task_name=obs.task_name,
        total_steps=obs.total_steps,
    )


@app.post("/step", response_model=StepResponse)
async def step(action: ActionRequest):
    """
    Submit an action and advance the environment by one step.

    Returns reward, whether episode is done, next observation, and grading details.
    """
    if env.is_done():
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call POST /reset to start a new episode.",
        )

    try:
        action_dict = {
            "bug_description": action.bug_description,
            "severity": action.severity,
            "suggested_fix": action.suggested_fix,
        }
        step_info = env.step(action_dict)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    next_obs = None
    if step_info.observation:
        next_obs = ObservationResponse(
            task_name=step_info.observation.task_name,
            code_snippet=step_info.observation.code_snippet,
            instructions=step_info.observation.instructions,
            step=step_info.observation.step,
            total_steps=step_info.observation.total_steps,
        )

    logger.info(
        f"Step completed — reward: {step_info.reward}, "
        f"done: {step_info.done}, "
        f"matched: {step_info.grade_result.matched_bugs}"
    )

    return StepResponse(
        reward=step_info.reward,
        done=step_info.done,
        observation=next_obs,
        reason=step_info.grade_result.reason,
        matched_bugs=step_info.grade_result.matched_bugs,
        cumulative_reward=step_info.cumulative_reward,
    )


@app.get("/observe", response_model=ObservationResponse)
async def observe():
    """Get the current observation without advancing the environment."""
    if env.is_done():
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )

    try:
        obs = env.observe()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ObservationResponse(
        task_name=obs.task_name,
        code_snippet=obs.code_snippet,
        instructions=obs.instructions,
        step=obs.step,
        total_steps=obs.total_steps,
    )


@app.get("/episode", response_model=EpisodeSummaryResponse)
async def episode_summary():
    """Get a summary of the current/completed episode."""
    summary = env.episode_summary()
    return EpisodeSummaryResponse(**summary)


@app.get("/list_tasks", response_model=List[TaskInfo])
async def list_tasks():
    """List all available tasks."""
    return [TaskInfo(**t) for t in env.list_tasks()]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting Code Review Environment server on port {port}")
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except OSError as e:
        # Port already in use — try alternative port (for validator environments)
        if "address already in use" in str(e).lower():
            alt_port = 8000
            logger.warning(f"Port {port} in use, trying {alt_port}...")
            uvicorn.run(app, host="0.0.0.0", port=alt_port)
        else:
            raise
