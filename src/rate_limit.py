"""
rate_limit.py -- Telling a rate limit that will clear from one that will not.

Groq returns 429 for two situations that need opposite responses:

  per-minute (RPM/TPM) -- clears in seconds. Sleep for the interval the provider
                          names in the error text, then retry. The call succeeds.
  per-day (RPD/TPD)    -- does not clear before tomorrow. Retrying is pure waste:
                          every attempt burns wall-clock time and returns the
                          same 429, and the budget does not refill mid-run.

The 2026-08-18 benchmark run is what this module exists to prevent. The
generator exhausted its 200,000 tokens/day at case ADV-04 and the runner carried
on regardless, issuing 39 more calls across the remaining 12 cases. Every one
returned 429 and was recorded as a pipeline error. The report was then generated
from 17 of 30 cases per arm, with the arms no longer scored on the same case
list -- while section 2.1 went on asserting they were.

The judge side already handled this (benchmark/evaluator.py stopped judging once
its own daily cap was gone); the generator side had nothing. This module holds
the shared logic so the two cannot drift apart again.

A rejected *credential* is a third case, and it clears on neither schedule: an
expired or revoked key fails every subsequent call until a human replaces it.
Run 20260819_082917 is what that costs when it is not detected -- the key in .env
had expired, all 90 generator calls returned 401, and the judge then made 30 more
calls to the same endpoint with the same key, retrying each one three times. The
report was generated anyway, from the one arm that never calls a model.

No provider SDK is imported here. The signal is the error text, because that is
all a LangChain-wrapped exception reliably carries.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

# Longest pause worth taking inline. Above this the limit is a daily one in all
# but name -- Groq expresses a spent daily budget as a retry delay of anywhere up
# to 24 hours -- and the run should stop rather than sleep through it.
DEFAULT_MAX_WAIT_SECS = 90.0

# "Please try again in 12m59.328s" / "try again in 7.5s"
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.I)

# Groq names the window it is enforcing. Matching it directly is more reliable
# than inferring from the retry delay, which is only a lower bound.
_PER_DAY_MARKERS = (
    "per day",
    "tokens per day",
    "requests per day",
    "(tpd)",
    "(rpd)",
)

# Credentials the provider refused. Matched on its wording *and* on the exception
# class LangChain surfaces, since only the message text survives the wrapping.
# 403 is included because Groq uses it for a key that is valid but disabled.
_AUTH_MARKERS = (
    "authenticationerror",
    "permissiondeniederror",
    "expired_api_key",
    "invalid_api_key",
    "invalid api key",
    "error code: 401",
    "error code: 403",
)


def max_rate_limit_wait() -> float:
    """Seconds to wait inline before treating a limit as unrecoverable."""
    try:
        return float(os.environ.get("LLM_MAX_RATE_LIMIT_WAIT", DEFAULT_MAX_WAIT_SECS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_WAIT_SECS


def retry_after_secs(message: str) -> Optional[float]:
    """
    Seconds the provider asks us to wait, parsed from its 429 text.

    None when the message carries no such hint -- which includes every error that
    is not a rate limit at all, so callers can use this as the rate-limit test.
    """
    match = _RETRY_AFTER_RE.search(message or "")
    if not match:
        return None
    minutes = float(match.group(1)) if match.group(1) else 0.0
    return minutes * 60.0 + float(match.group(2))


def is_daily_quota(message: str, wait_secs: Optional[float] = None) -> bool:
    """
    True when the 429 is a spent daily budget rather than a momentary burst.

    Two independent signals, because neither is guaranteed: the window Groq
    names in the message, and a retry delay too long to sit through. A provider
    that words its message differently still trips the second test.
    """
    text = (message or "").lower()
    if any(marker in text for marker in _PER_DAY_MARKERS):
        return True
    if wait_secs is None:
        wait_secs = retry_after_secs(message)
    return wait_secs is not None and wait_secs > max_rate_limit_wait()


def is_auth_failure(message: str) -> bool:
    """
    True when the provider rejected our credentials rather than our request rate.

    Kept apart from `is_daily_quota` because the two demand the same immediate
    response -- stop calling -- for opposite reasons. A spent daily budget refills
    overnight, so the run is worth resuming. A dead key is worth nothing until
    someone edits .env, and telling a user to "try again after the reset" when
    their key has expired sends them to wait for something that will never happen.
    """
    text = (message or "").lower()
    return any(marker in text for marker in _AUTH_MARKERS)


class CredentialState:
    """
    Process-wide latch for credentials the provider refused.

    Deliberately ONE instance rather than one per role, which is where this
    differs from QuotaState: the generator and the judge hold separate daily
    budgets but authenticate with the same GROQ_API_KEY. So the judge running dry
    says nothing about the generator, while the generator's key being dead is
    proof the judge's calls cannot succeed either.

    No retry_after: there is no interval that fixes this.
    """

    def __init__(self) -> None:
        self.hit = False
        self.detail = ""

    def record(self, message: str) -> None:
        if self.hit:
            return
        self.hit = True
        self.detail = (message or "")[:300]

    def reset(self) -> None:
        """For tests, and for a run started after the key was replaced."""
        self.hit = False
        self.detail = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"hit": self.hit, "detail": self.detail}


class QuotaState:
    """
    Process-wide latch: once a daily budget is gone, it stays gone for this run.

    Deliberately a latch and not a counter. The first caller to see the daily 429
    records it, and every later caller checks `hit` and skips instead of
    discovering the same thing at the cost of another round trip. One instance
    per model role, since the generator and judge are different models with
    separate budgets -- the judge running dry says nothing about the generator.
    """

    def __init__(self, role: str) -> None:
        self.role = role
        self.hit = False
        self.detail = ""
        self.retry_after_secs: Optional[float] = None

    def record(self, message: str, wait_secs: Optional[float] = None) -> None:
        if self.hit:
            return
        self.hit = True
        self.detail = (message or "")[:300]
        self.retry_after_secs = (
            wait_secs if wait_secs is not None else retry_after_secs(message)
        )

    def reset(self) -> None:
        """For tests, and for a resumed run that starts a fresh budget."""
        self.hit = False
        self.detail = ""
        self.retry_after_secs = None

    def human_wait(self) -> str:
        """The provider's retry delay as something readable in a console line."""
        secs = self.retry_after_secs
        if secs is None:
            return "unknown"
        if secs < 60:
            return f"{secs:.0f}s"
        if secs < 3600:
            return f"{secs / 60:.0f}m"
        return f"{secs / 3600:.1f}h"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "hit": self.hit,
            "detail": self.detail,
            "retry_after_secs": self.retry_after_secs,
        }


# One latch per role. Imported by both the generation node and the evaluator so
# a run has a single view of what is exhausted.
GENERATOR_QUOTA = QuotaState("generator")
JUDGE_QUOTA = QuotaState("judge")

# One latch for the credential, shared by both roles -- see CredentialState.
AUTH_FAILED = CredentialState()
