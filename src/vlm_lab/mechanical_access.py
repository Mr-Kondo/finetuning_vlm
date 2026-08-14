"""All-split access for mechanical, content-free checks (ADR-008).

Two callers outside Phase 5 legitimately have to touch test-split rows without
evaluating them: the ADR-008/ADR-019 duplication audit, which must hash them to
find train/validation duplicates, and the ADR-008 mechanical parse-count check in
``notebooks/01_dataset.ipynb``, which counts parse successes and nothing else.
Both are *counts and hashes only* — no content is read, displayed, or scored.

This lives in its own module for the same reason :mod:`vlm_lab.sealed_test` does
(review findings F5, R2-12, R3-14): a function that can reach all three splits
sitting inside :mod:`vlm_lab.data` would be visible to every Phase-2 module that
imports `vlm_lab.data`, which is the footgun those findings are about. Keeping it
here leaves `vlm_lab.data` with no path to the test split at all.

**Honest limit of this boundary.** Python has no capability enforcement: a
determined caller can import any module. What module structure buys is that the
test split is not reachable *by accident* from the development loader, and that
every deliberate reach is named, logged, and obvious in review. The access log,
not the import graph, is the durable evidence.

**Rows returned here may be used to compute hashes, cluster structure and
aggregate counts only** — never to inspect, display, or evaluate test-split
images or ground truth. Per proposal §8.1 the audit publishes aggregate counts
and a frozen verdict, never hashes or row IDs, because a test-side hash can be
joined against an already-viewed train image to reveal test content.
"""

from __future__ import annotations

from pathlib import Path

import datasets

from vlm_lab.access_log import record_split_access
from vlm_lab.data import ALL_SPLIT_NAMES, load_named_splits

__all__ = ["load_all_splits_for_mechanical_check"]


def load_all_splits_for_mechanical_check(
    purpose: str,
    revision: str | None = None,
    access_log_path: Path | None = None,
) -> datasets.DatasetDict:
    """Load all three CORD v2 splits for a mechanical, content-free check.

    `purpose` is required and is written into the access log, so the record
    says *why* the test split was touched rather than only that it was. Use a
    stable identifier such as ``"adr-008-duplication-audit"`` or
    ``"adr-008-mechanical-parse-count"``.

    Every invocation is appended to the access log before any split is
    requested; see :func:`vlm_lab.access_log.record_split_access` for the
    record's fields and for `access_log_path`'s default.

    ``revision`` is passed through to ``datasets.load_dataset`` untouched;
    ``None`` follows the Hub's default revision.
    """
    record_split_access(
        "load_all_splits_for_mechanical_check",
        log_path=access_log_path,
        purpose=purpose,
    )
    return load_named_splits(ALL_SPLIT_NAMES, revision)
