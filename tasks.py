"""
tasks.py — Task definitions for the Code Review RL Environment.

Each task contains:
    - A realistic code snippet with intentional bugs / vulnerabilities.
    - Step-by-step instructions the agent must follow.
    - Ground-truth labels used by the grading system.

Difficulty levels:
    EASY   — single-function, single bug.
    MEDIUM — multi-function backend code, 3 distinct issues.
    HARD   — full pull-request review with security, testing, and design issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class GroundTruthBug:
    """A single expected bug the agent should detect."""
    bug_keywords: List[str]           # Any keyword match counts as detection
    expected_severity: Severity
    fix_keywords: List[str]           # Keywords expected in suggested fix
    description: str                  # Human-readable summary (for README / logs)
    weight: float = 1.0              # Relative weight in partial-credit grading


@dataclass(frozen=True)
class TaskStep:
    """One step the agent must complete."""
    instruction: str
    ground_truth_bugs: List[GroundTruthBug]


@dataclass(frozen=True)
class Task:
    """Complete task definition."""
    task_id: str
    task_name: str
    difficulty: Difficulty
    code_snippet: str
    steps: List[TaskStep]
    max_steps: int = 0               # 0 → len(steps)

    def __post_init__(self):
        if self.max_steps == 0:
            object.__setattr__(self, "max_steps", len(self.steps))


# ---------------------------------------------------------------------------
# EASY TASK — Off-by-one error in a utility function
# ---------------------------------------------------------------------------

EASY_CODE = '''\
def find_max_subarray_sum(nums: list[int], k: int) -> int:
    """Return the maximum sum of any contiguous subarray of length k."""
    if not nums or k <= 0:
        return 0

    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(1, len(nums) - k):  # BUG: should be len(nums) - k + 1
        window_sum += nums[i + k - 1] - nums[i - 1]
        max_sum = max(max_sum, window_sum)

    return max_sum
'''

EASY_TASK = Task(
    task_id="easy_offbyone",
    task_name="Sliding Window Off-by-One",
    difficulty=Difficulty.EASY,
    code_snippet=EASY_CODE,
    steps=[
        TaskStep(
            instruction=(
                "Review the function `find_max_subarray_sum`. "
                "Identify the bug, assign a severity level, and suggest a fix. "
                "Respond with JSON keys: bug_description, severity, suggested_fix."
            ),
            ground_truth_bugs=[
                GroundTruthBug(
                    bug_keywords=["off-by-one", "off by one", "range", "len(nums) - k + 1",
                                  "boundary", "last subarray", "missing last", "fence post",
                                  "fencepost", "index", "iteration"],
                    expected_severity=Severity.MEDIUM,
                    fix_keywords=["len(nums) - k + 1", "range(1, len(nums) - k + 1)",
                                  "<= len(nums) - k", "inclusive"],
                    description="Off-by-one in loop range skips the last valid subarray window.",
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# MEDIUM TASK — Authentication module with 3 bugs
# ---------------------------------------------------------------------------

MEDIUM_CODE = '''\
import sqlite3
import hashlib

DATABASE = "users.db"


def get_db():
    return sqlite3.connect(DATABASE)


def authenticate_user(username: str, password: str) -> dict | None:
    """Authenticate a user and return their profile."""
    db = get_db()
    cursor = db.cursor()

    # BUG 1: SQL injection — string formatting instead of parameterised query
    query = f"SELECT id, username, role, password_hash FROM users WHERE username = '{username}'"
    cursor.execute(query)
    row = cursor.fetchone()

    if row is None:
        return None

    stored_hash = row[3]
    # BUG 2: Using MD5 without salt — weak hashing
    if hashlib.md5(password.encode()).hexdigest() != stored_hash:
        return None

    return {"id": row[0], "username": row[1], "role": row[2]}


def get_user_profile(requesting_user_id: int, target_user_id: int) -> dict | None:
    """Fetch another user's profile. Should enforce permissions."""
    db = get_db()
    cursor = db.cursor()

    # BUG 3: No permission check — any authenticated user can view any profile
    cursor.execute("SELECT id, username, email, role FROM users WHERE id = ?", (target_user_id,))
    row = cursor.fetchone()

    if row is None:
        return None

    return {"id": row[0], "username": row[1], "email": row[2], "role": row[3]}


def delete_user(admin_id: int, target_user_id: int) -> bool:
    """Delete a user account. Only admins should be able to do this."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (target_user_id,))
    db.commit()
    return cursor.rowcount > 0
'''

MEDIUM_TASK = Task(
    task_id="medium_auth",
    task_name="Authentication Module Review",
    difficulty=Difficulty.MEDIUM,
    code_snippet=MEDIUM_CODE,
    steps=[
        # Step 1 — SQL Injection
        TaskStep(
            instruction=(
                "Step 1/3: Review `authenticate_user` for security vulnerabilities. "
                "Focus on how the database query is constructed. "
                "Respond with JSON keys: bug_description, severity, suggested_fix."
            ),
            ground_truth_bugs=[
                GroundTruthBug(
                    bug_keywords=["sql injection", "sql-injection", "string format",
                                  "f-string", "unsanitized", "parameterized",
                                  "parameterised", "interpolation", "user input"],
                    expected_severity=Severity.CRITICAL,
                    fix_keywords=["parameterized", "parameterised", "placeholder", "?",
                                  "execute(", "bind", "prepared statement"],
                    description="SQL injection via f-string query construction.",
                    weight=1.5,
                ),
            ],
        ),
        # Step 2 — Weak hashing
        TaskStep(
            instruction=(
                "Step 2/3: Continue reviewing `authenticate_user`. "
                "Examine how passwords are verified. "
                "Respond with JSON keys: bug_description, severity, suggested_fix."
            ),
            ground_truth_bugs=[
                GroundTruthBug(
                    bug_keywords=["md5", "weak hash", "no salt", "unsalted",
                                  "insecure hash", "hashing", "rainbow table",
                                  "brute force", "collision"],
                    expected_severity=Severity.HIGH,
                    fix_keywords=["bcrypt", "argon2", "scrypt", "pbkdf2", "salt",
                                  "passlib", "hashlib.sha256", "sha-256"],
                    description="MD5 without salt is cryptographically weak for passwords.",
                    weight=1.0,
                ),
            ],
        ),
        # Step 3 — Permission bypass
        TaskStep(
            instruction=(
                "Step 3/3: Review `get_user_profile` and `delete_user` for authorisation issues. "
                "Check if proper access control is enforced. "
                "Respond with JSON keys: bug_description, severity, suggested_fix."
            ),
            ground_truth_bugs=[
                GroundTruthBug(
                    bug_keywords=["permission", "authorization", "authorisation",
                                  "access control", "privilege", "IDOR", "insecure direct",
                                  "no check", "any user", "bypass", "role check",
                                  "admin check", "ownership"],
                    expected_severity=Severity.HIGH,
                    fix_keywords=["check role", "verify permission", "admin", "owner",
                                  "requesting_user", "role == 'admin'", "authorization",
                                  "authorisation", "access control", "rbac"],
                    description="Missing permission checks allow horizontal/vertical privilege escalation.",
                    weight=1.2,
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# HARD TASK — Full pull-request review
# ---------------------------------------------------------------------------

HARD_CODE = '''\
# PR #482: Add payment processing endpoint
# Author: dev-intern
# Branch: feature/payment-gateway

import os
import json
import logging
import requests
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)
logger = logging.getLogger(__name__)

PAYMENT_API_KEY = os.getenv("PAYMENT_API_KEY", "sk_test_default_key_12345")
PAYMENT_GATEWAY_URL = "https://api.paymentgateway.com/v1/charges"


def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Missing token"}), 401
        # BUG 1: Token is never actually validated — just checks existence
        return f(*args, **kwargs)
    return decorated


class PaymentProcessor:
    """Handle payment processing logic."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache = {}

    def process_payment(self, user_id: str, amount: float, currency: str) -> dict:
        """Process a payment through the external gateway."""
        # BUG 2: No input validation — negative amounts, unsupported currencies
        payload = {
            "amount": int(amount * 100),  # Convert to cents
            "currency": currency,
            "source": user_id,
            "api_key": self.api_key,      # BUG 3: API key sent in payload body
        }

        try:
            # BUG 4: No timeout on external request — can hang indefinitely
            response = requests.post(PAYMENT_GATEWAY_URL, json=payload)
            result = response.json()
        except Exception as e:
            # BUG 5: Broad exception catch, logs full payload including API key
            logger.error(f"Payment failed: {e}, payload: {payload}")
            return {"status": "error", "message": str(e)}

        # Cache the transaction (no TTL or size limit)
        self._cache[user_id] = result
        return result

    def get_transaction_history(self, user_id: str) -> list:
        """Retrieve transaction history for a user."""
        # BUG 6: Returns cached data only — not fetching from DB, stale data
        return self._cache.get(user_id, [])


processor = PaymentProcessor(PAYMENT_API_KEY)


@app.route("/api/v1/pay", methods=["POST"])
@require_auth
def create_payment():
    """Create a new payment."""
    data = request.get_json()

    # BUG 7: No schema validation on incoming request body
    user_id = data.get("user_id")
    amount = data.get("amount")
    currency = data.get("currency", "usd")

    result = processor.process_payment(user_id, amount, currency)

    # BUG 8: Returning full internal result to client — may leak sensitive info
    return jsonify(result), 200


@app.route("/api/v1/transactions/<user_id>", methods=["GET"])
@require_auth
def get_transactions(user_id):
    """Get transaction history."""
    # BUG 9: No check that authenticated user owns these transactions (IDOR)
    history = processor.get_transaction_history(user_id)
    return jsonify({"transactions": history}), 200


@app.route("/api/v1/refund", methods=["POST"])
@require_auth
def process_refund():
    """Process a refund — not yet implemented."""
    # BUG 10: Stub endpoint returns 200 OK with no logic — misleading
    return jsonify({"status": "refund_processed"}), 200


if __name__ == "__main__":
    # BUG 11: Debug mode in production code
    app.run(debug=True, host="0.0.0.0", port=8080)
'''

HARD_TASK = Task(
    task_id="hard_pr_review",
    task_name="Payment Gateway PR Review",
    difficulty=Difficulty.HARD,
    code_snippet=HARD_CODE,
    steps=[
        # Step 1 — Security vulnerabilities
        TaskStep(
            instruction=(
                "Step 1/3: Perform a security audit of this pull request. "
                "Identify ALL security vulnerabilities in the authentication, "
                "payment processing, and data exposure areas. "
                "Respond with JSON keys: bug_description, severity, suggested_fix."
            ),
            ground_truth_bugs=[
                GroundTruthBug(
                    bug_keywords=["token", "not validated", "never verified",
                                  "authentication bypass", "jwt", "no verification",
                                  "existence check", "token validation"],
                    expected_severity=Severity.CRITICAL,
                    fix_keywords=["validate token", "verify jwt", "decode token",
                                  "token verification", "signature"],
                    description="Auth decorator only checks token existence, never validates it.",
                    weight=1.5,
                ),
                GroundTruthBug(
                    bug_keywords=["api_key", "api key", "payload", "body",
                                  "credential", "secret", "exposed",
                                  "sensitive", "key in payload"],
                    expected_severity=Severity.CRITICAL,
                    fix_keywords=["header", "Authorization header", "Bearer",
                                  "environment variable", "secret management",
                                  "do not send", "remove from payload"],
                    description="API key sent in request payload body instead of auth header.",
                    weight=1.5,
                ),
                GroundTruthBug(
                    bug_keywords=["IDOR", "insecure direct", "user_id", "ownership",
                                  "authorization", "authorisation", "any user",
                                  "access control", "transaction"],
                    expected_severity=Severity.HIGH,
                    fix_keywords=["check ownership", "verify user", "authenticated user",
                                  "session user", "match user_id", "authorization"],
                    description="IDOR in transaction history — no ownership verification.",
                    weight=1.0,
                ),
                GroundTruthBug(
                    bug_keywords=["log", "payload", "api key", "sensitive",
                                  "credential leak", "error log", "exception"],
                    expected_severity=Severity.HIGH,
                    fix_keywords=["redact", "sanitize", "mask", "remove sensitive",
                                  "structured logging", "do not log"],
                    description="Error handler logs full payload including API key.",
                    weight=0.8,
                ),
                GroundTruthBug(
                    bug_keywords=["debug", "debug=True", "production",
                                  "development mode", "werkzeug debugger"],
                    expected_severity=Severity.HIGH,
                    fix_keywords=["debug=False", "environment variable",
                                  "production config", "FLASK_ENV",
                                  "remove debug", "gunicorn"],
                    description="Flask debug mode enabled in production code.",
                    weight=0.6,
                ),
            ],
        ),
        # Step 2 — Missing edge cases / tests
        TaskStep(
            instruction=(
                "Step 2/3: Identify missing input validations, edge cases, "
                "and error handling gaps. What test cases are missing? "
                "Respond with JSON keys: bug_description, severity, suggested_fix."
            ),
            ground_truth_bugs=[
                GroundTruthBug(
                    bug_keywords=["validation", "negative amount", "zero amount",
                                  "currency", "input validation", "schema",
                                  "type check", "None", "missing field"],
                    expected_severity=Severity.HIGH,
                    fix_keywords=["validate", "pydantic", "marshmallow", "schema",
                                  "check amount", "positive", "> 0", "whitelist",
                                  "allowed currencies"],
                    description="No input validation on payment amount or currency.",
                    weight=1.2,
                ),
                GroundTruthBug(
                    bug_keywords=["timeout", "requests.post", "hang", "indefinite",
                                  "no timeout", "connection", "retry"],
                    expected_severity=Severity.MEDIUM,
                    fix_keywords=["timeout=", "retry", "circuit breaker",
                                  "requests.post(url, timeout", "max_retries"],
                    description="External HTTP request has no timeout or retry logic.",
                    weight=0.8,
                ),
                GroundTruthBug(
                    bug_keywords=["refund", "stub", "not implemented",
                                  "returns 200", "misleading", "fake",
                                  "no logic", "placeholder"],
                    expected_severity=Severity.MEDIUM,
                    fix_keywords=["501", "not implemented", "raise",
                                  "remove endpoint", "implement", "TODO",
                                  "return 501"],
                    description="Refund endpoint returns 200 OK but does nothing.",
                    weight=0.6,
                ),
                GroundTruthBug(
                    bug_keywords=["broad exception", "bare except",
                                  "Exception as e", "generic exception",
                                  "catch all", "swallow"],
                    expected_severity=Severity.MEDIUM,
                    fix_keywords=["specific exception", "RequestException",
                                  "ConnectionError", "Timeout", "HTTPError",
                                  "narrow exception"],
                    description="Overly broad exception handling hides real errors.",
                    weight=0.5,
                ),
            ],
        ),
        # Step 3 — Design review comment
        TaskStep(
            instruction=(
                "Step 3/3: Write a professional code review summary comment "
                "for this PR. Cover design concerns, architectural improvements, "
                "and overall recommendation (approve / request changes). "
                "Respond with JSON keys: bug_description, severity, suggested_fix."
            ),
            ground_truth_bugs=[
                GroundTruthBug(
                    bug_keywords=["cache", "no TTL", "no eviction", "memory leak",
                                  "unbounded", "stale data", "in-memory",
                                  "_cache", "size limit"],
                    expected_severity=Severity.MEDIUM,
                    fix_keywords=["redis", "TTL", "LRU", "eviction policy",
                                  "database", "expiration", "max size",
                                  "cachetools", "functools.lru_cache"],
                    description="In-memory cache with no TTL or size limit causes staleness and leaks.",
                    weight=1.0,
                ),
                GroundTruthBug(
                    bug_keywords=["default key", "sk_test_default", "hardcoded",
                                  "fallback key", "default api key",
                                  "getenv default", "secret in code"],
                    expected_severity=Severity.HIGH,
                    fix_keywords=["remove default", "require env var", "raise error",
                                  "no fallback", "secret manager", "vault",
                                  "mandatory environment"],
                    description="Hardcoded default API key used as fallback in production.",
                    weight=1.0,
                ),
                GroundTruthBug(
                    bug_keywords=["request changes", "not ready", "do not merge",
                                  "needs work", "block", "reject",
                                  "significant issues", "major concerns"],
                    expected_severity=Severity.CRITICAL,
                    fix_keywords=["request changes", "address", "before merging",
                                  "security review", "follow-up", "fix before",
                                  "re-review"],
                    description="PR should be blocked — too many critical security issues.",
                    weight=0.8,
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, Task] = {
    EASY_TASK.task_id: EASY_TASK,
    MEDIUM_TASK.task_id: MEDIUM_TASK,
    HARD_TASK.task_id: HARD_TASK,
}


def get_task(task_id: str) -> Task:
    """Retrieve a task by ID, raising KeyError if not found."""
    if task_id not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task_id '{task_id}'. "
            f"Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]


def list_tasks() -> list[dict]:
    """Return a summary list of all available tasks."""
    return [
        {
            "task_id": t.task_id,
            "task_name": t.task_name,
            "difficulty": t.difficulty.value,
            "num_steps": len(t.steps),
        }
        for t in TASK_REGISTRY.values()
    ]
