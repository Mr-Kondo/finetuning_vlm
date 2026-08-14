"""Append-only access log for dataset loads that can reach the held-out test split.

`EXPERIMENT_SPEC.md` §8b allows exactly one test-split execution. That claim is
only auditable if every load that can reach `test` leaves a durable trace, so
:func:`record_split_access` is called by :func:`vlm_lab.sealed_test.load_sealed_test_split`
and :func:`vlm_lab.mechanical_access.load_all_splits_for_mechanical_check`.

The development loader deliberately does *not* log: it cannot reach `test` at all,
so there is nothing to audit.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ACCESS_LOG_PATH = Path("logs/split_access.jsonl")

# Frames above record_split_access: 1 = the loader that called it,
# 2 = the module that called the loader, which is the one worth recording.
_LOADER_CALLER_FRAME_DEPTH = 2

_GIT_COMMAND_TIMEOUT_SECONDS = 5


def record_split_access(
    function_name: str,
    log_path: Path | None = None,
    purpose: str | None = None,
) -> None:
    """Append one JSON line recording an invocation of a test-reachable loader.

    Writes a `timestamp_utc` / `function` / `caller_module` / `git_commit` /
    `purpose` record to `log_path` (default :data:`DEFAULT_ACCESS_LOG_PATH`), creating the
    parent directory if needed.

    A logging failure must not abort the load it is recording, so write errors
    become a `UserWarning` rather than an exception. They are never silently
    discarded.
    """
    resolved_log_path = Path(log_path) if log_path is not None else DEFAULT_ACCESS_LOG_PATH
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "function": function_name,
        "caller_module": _module_name_above(_LOADER_CALLER_FRAME_DEPTH),
        "git_commit": _current_git_commit(),
        "purpose": purpose,
    }

    try:
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry) + "\n")
    except OSError as exc:
        warnings.warn(
            f"Could not append to the split access log at {resolved_log_path}: {exc}",
            stacklevel=2,
        )


def _module_name_above(frame_depth: int) -> str | None:
    """Return `__name__` of the module `frame_depth` frames above this helper's caller.

    Returns ``None`` when the stack is shallower than requested or when the
    interpreter does not expose frames.
    """
    frame = inspect.currentframe()
    if frame is None:
        return None

    frame = frame.f_back  # the caller of this helper: record_split_access
    for _ in range(frame_depth):
        if frame is None:
            return None
        frame = frame.f_back

    if frame is None:
        return None
    return frame.f_globals.get("__name__")


def _current_git_commit() -> str | None:
    """Return the current git commit hash, or ``None`` if it cannot be obtained.

    An installed copy of this package need not sit inside a git checkout, and
    `git` need not be present at all, so every failure mode degrades to ``None``
    rather than breaking the load being logged.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=_GIT_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None
