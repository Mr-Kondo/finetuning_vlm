"""Sealed access to the CORD v2 held-out test split. **Phase 5 only.**

This module exists to make the test split unreachable from Phase-2 code rather
than merely discouraged there (review finding R3-14): separately-named functions
inside one module are not a capability boundary, because anything that imports
:mod:`vlm_lab.data` can still see every name in it. So the sealed loader lives
here instead, and :mod:`vlm_lab.data` does not import this module — a Phase-2
module that stays within `vlm_lab.data` cannot reach the test split by any path.

Importing this module is itself the signal that Phase 5 has begun. Nothing in
Phases 1-4, including `notebooks/02_baseline.ipynb` and the training code, may
import it. Phases 1-4 use :func:`vlm_lab.data.load_development_splits`; the
ADR-008 duplication audit uses
:func:`vlm_lab.duplication_audit.load_for_duplication_audit`, which reads
test rows for hashing only.
"""

from __future__ import annotations

from pathlib import Path

import datasets

from vlm_lab.access_log import record_split_access
from vlm_lab.data import CORD_V2_REPO_ID

SEALED_TEST_SPLIT_NAME = "test"


def load_sealed_test_split(
    revision: str | None = None,
    access_log_path: Path | None = None,
) -> datasets.Dataset:
    """Load the held-out CORD v2 test split for the final Phase 5 comparison.

    Every invocation is appended to the access log before the split is
    requested, so that `EXPERIMENT_SPEC.md` §8b's "one test execution" is
    auditable from a durable artifact rather than asserted. See
    :func:`vlm_lab.access_log.record_split_access` for the record's fields and
    for `access_log_path`'s default.

    ``revision`` is passed through to ``datasets.load_dataset`` untouched;
    ``None`` follows the Hub's default revision.
    """
    record_split_access("load_sealed_test_split", log_path=access_log_path)
    return datasets.load_dataset(
        CORD_V2_REPO_ID, split=SEALED_TEST_SPLIT_NAME, revision=revision
    )
