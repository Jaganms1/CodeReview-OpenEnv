---
title: Code Review Env
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🔍 Code Review Environment for LLM Agents

**An OpenEnv-compatible reinforcement learning environment where AI agents learn to review code like senior engineers.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Overview

This project simulates real-world **pull request review workflows** as a reinforcement learning environment. An LLM agent acts as a code reviewer — receiving code snippets, identifying bugs, assigning severity levels, and suggesting fixes. The environment grades each response with a deterministic reward signal, enabling RL training loops or evaluation benchmarks.

### Why This Matters

Code review is one of the highest-leverage activities in software engineering, yet it's cognitively demanding and inconsistent across teams. By framing it as an RL problem, we can:

- **Benchmark** LLM code understanding capabilities
- **Train** agents that give consistent, high-quality reviews
- **Evaluate** reasoning about security, correctness, and design

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    inference.py                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │  OpenAI   │◄──│  RL Loop  │──►│  CodeReviewEnv   │   │
│  │  Client   │    │          │    │    (env.py)       │   │
│  └──────────┘    └──────────┘    └────────┬─────────┘   │
│                                           │              │
│                       ┌───────────────────┼──────────┐   │
│                       │                   ▼          │   │
│                  ┌────┴─────┐     ┌──────────────┐   │   │
│                  │ tasks.py  │     │  graders.py   │   │   │
│                  │          │     │               │   │   │
│                  │ EASY     │     │ Detection     │   │   │
│                  │ MEDIUM   │     │ Severity      │   │   │
│                  │ HARD     │     │ Fix Quality   │   │   │
│                  └──────────┘     └──────────────┘   │   │
│                       └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### RL Loop

```
Agent                          Environment
  │                                │
  │◄─── obs = env.reset(task) ─────│
  │                                │
  ├──── action (JSON) ────────────►│
  │                                ├── grade_action()
  │◄─── reward + next_obs ─────────│
  │                                │
  ├──── action (JSON) ────────────►│
  │          ...                   │
  │◄─── done=True, final_reward ───│
  │                                │
```

---

## 📂 Project Structure

```
code_review_env/
├── inference.py          # Main entry point — runs the RL loop with OpenAI client
├── env.py                # OpenEnv-compatible environment (reset / step / observe)
├── tasks.py              # Task definitions: EASY, MEDIUM, HARD
├── graders.py            # Deterministic keyword-based grading engine
├── openenv.yaml          # OpenEnv environment configuration
├── Dockerfile            # Production-ready container
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🎯 Tasks

### 1. Easy — Sliding Window Off-by-One

| Property     | Value |
|:------------|:------|
| **Steps**   | 1     |
| **Bug**     | Off-by-one in loop range skips the last valid subarray |
| **Skills**  | Algorithm correctness, boundary analysis |
| **Grading** | Binary + step penalty |

### 2. Medium — Authentication Module Review

| Property     | Value |
|:------------|:------|
| **Steps**   | 3     |
| **Bugs**    | SQL injection, MD5 without salt, missing permission checks |
| **Skills**  | Security analysis, authentication best practices |
| **Grading** | Partial credit per step |

### 3. Hard — Payment Gateway PR Review

| Property     | Value |
|:------------|:------|
| **Steps**   | 3     |
| **Bugs**    | 11+ issues: auth bypass, IDOR, credential exposure, no validation, debug mode, etc. |
| **Skills**  | Full security audit, test gap analysis, design review |
| **Grading** | Weighted partial credit with multi-bug detection |

---

## 📐 Observation & Action Formats

### Observation (what the agent sees)

```json
{
  "task_name": "Payment Gateway PR Review",
  "code_snippet": "import os\nimport json\n...",
  "instructions": "Step 1/3: Perform a security audit...",
  "step": 1,
  "total_steps": 3
}
```

### Action (what the agent returns)

```json
{
  "bug_description": "SQL injection vulnerability via f-string query construction in authenticate_user. User-controlled input is directly interpolated into the SQL query without sanitization.",
  "severity": "critical",
  "suggested_fix": "Use parameterized queries with placeholder syntax: cursor.execute('SELECT ... WHERE username = ?', (username,))"
}
```

---

## 🏆 Reward Strategy

### Components

| Component          | Weight | Description |
|:-------------------|:-------|:------------|
| Bug Detection      | 50%    | Keyword match against ground-truth bug descriptions |
| Severity Accuracy  | 20%    | Distance-based scoring (exact=1.0, ±1=0.5, ±2=0.2) |
| Fix Relevance      | 30%    | Keyword match on suggested fix quality |

### Step Penalty

A penalty of **0.02 per step** is applied to encourage efficient convergence. The penalty accumulates with each step.

### Scoring Example

```
Bug detected:      ✅ (0.50 × 1.0 = 0.50)
Severity correct:  ✅ (0.20 × 1.0 = 0.20)
Fix relevant:      ✅ (0.30 × 1.0 = 0.30)
Step penalty:      -0.02 (step 1)
────────────────────────────
Total reward:      0.98
```

### Multi-Bug Steps

When a step has multiple ground-truth bugs (e.g., hard task step 1), scores are **weight-averaged** across all expected bugs. Detecting more bugs yields a higher score.

---

## 🚀 Setup & Usage

### Prerequisites

- Python 3.11+
- An OpenAI-compatible API endpoint
- `HF_TOKEN` environment variable set

### Local Setup

```bash
# Clone and navigate
cd code_review_env

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your_huggingface_token"
export API_BASE_URL="https://api.openai.com/v1"  # or your endpoint
export MODEL_NAME="gpt-4o"                        # or your model

# Run inference
python inference.py
```

### Docker

```bash
# Build
docker build -t code-review-env .

# Run
docker run --rm \
  -e HF_TOKEN="your_token" \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o" \
  code-review-env
```

### Hugging Face Spaces

1. Push the repository to a Hugging Face Space
2. Set `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME` as Space secrets
3. The Dockerfile will handle the rest

---

## 📤 Output Format

The inference script produces output in the **exact** format required by OpenEnv:

```
[START]
[STEP] {"task_id": "easy_offbyone", "step": 1, "reward": 0.98, "done": true, ...}
[STEP] {"episode_summary": {"task_id": "easy_offbyone", "episode_reward": 0.98, ...}}
[STEP] {"task_id": "medium_auth", "step": 1, "reward": 0.85, "done": false, ...}
[STEP] {"task_id": "medium_auth", "step": 2, "reward": 0.72, "done": false, ...}
[STEP] {"task_id": "medium_auth", "step": 3, "reward": 0.90, "done": true, ...}
[STEP] {"episode_summary": {"task_id": "medium_auth", "episode_reward": 0.8233, ...}}
[STEP] {"task_id": "hard_pr_review", "step": 1, "reward": 0.76, "done": false, ...}
...
[END]
```

---

## 🔧 Using the Environment Programmatically

```python
from env import CodeReviewEnv

env = CodeReviewEnv()

# List available tasks
print(env.list_tasks())

# Run an episode
obs = env.reset("medium_auth")

while not env.is_done():
    print(f"Step {obs.step}/{obs.total_steps}: {obs.instructions}")

    action = {
        "bug_description": "SQL injection via f-string interpolation",
        "severity": "critical",
        "suggested_fix": "Use parameterized queries with ? placeholders"
    }

    step_info = env.step(action)
    print(f"  Reward: {step_info.reward}")
    print(f"  Reason: {step_info.grade_result.reason}")

    if step_info.observation:
        obs = step_info.observation

# Episode complete
print(f"\nFinal reward: {env.episode_reward()}")
print(env.episode_summary())
```

---

## 🧪 Grading Deep Dive

### Keyword Matching

The grader uses **case-insensitive keyword matching** against predefined ground-truth labels. For bug detection:

```python
# Ground truth for SQL injection
bug_keywords = ["sql injection", "sql-injection", "string format",
                "f-string", "unsanitized", "parameterized"]

# Agent says: "There's a SQL injection vulnerability due to f-string usage"
# → Matches "sql injection" AND "f-string" → detection_score = 1.0
```

### Severity Distance

Severity is scored on a **distance scale**, not binary match:

| Predicted vs Expected | Score |
|:---------------------|:------|
| Exact match           | 1.0   |
| 1 level off           | 0.5   |
| 2 levels off          | 0.2   |
| 3 levels off          | 0.0   |

### Weighted Multi-Bug Scoring

Each ground-truth bug has a `weight` that reflects its importance:

```python
# Critical SQL injection: weight = 1.5
# Missing permission check: weight = 1.2
# Weak hashing: weight = 1.0
# Total weight = 3.7

# If agent detects SQL injection + weak hashing:
# detection_score = (1.5 + 1.0) / 3.7 = 0.676
```

---

## 📊 Evaluation Metrics

When running across all tasks, the system tracks:

- **Per-step reward** — How well the agent performed on each individual step
- **Per-task episode reward** — Mean reward across all steps of a task
- **Overall score** — Aggregate across all tasks
- **Bug detection rate** — Fraction of ground-truth bugs correctly identified
- **Severity accuracy** — How close severity predictions are to ground truth
- **Fix quality** — Relevance of suggested fixes

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Acknowledgements

Built for the **Meta PyTorch OpenEnv Hackathon**. Designed to push the boundaries of LLM evaluation in software engineering tasks.
