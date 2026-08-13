"""Unit tests for src/vlm_lab/mechanical_access.py's all-split access path.

This is the only non-Phase-5 path to the held-out test split, and it may only
count or hash it (ADR-008, ADR-019). These tests pin two properties:
it really does request all three splits, and every invocation lands in the
access log — which is the durable evidence that a deliberate test read
happened, since Python cannot enforce the boundary itself.
"""

from __future__ import annotations

import json

from vlm_lab.data import ALL_SPLIT_NAMES
from vlm_lab.mechanical_access import load_all_splits_for_mechanical_check


def test_mechanical_check_requests_all_three_splits(
    recorded_hub_split_requests, tmp_path
):
    load_all_splits_for_mechanical_check(
        "adr-008-duplication-audit", access_log_path=tmp_path / "split_access.jsonl"
    )

    requested_splits = [call["split"] for call in recorded_hub_split_requests]
    assert requested_splits == list(ALL_SPLIT_NAMES)


def test_mechanical_check_appends_an_access_log_entry_naming_its_purpose(
    recorded_hub_split_requests, tmp_path
):
    access_log_path = tmp_path / "split_access.jsonl"

    load_all_splits_for_mechanical_check(
        "adr-008-duplication-audit", access_log_path=access_log_path
    )

    entry = json.loads(access_log_path.read_text(encoding="utf-8").strip())
    assert entry["function"] == "load_all_splits_for_mechanical_check"
    assert entry["purpose"] == "adr-008-duplication-audit"
    assert entry["caller_module"] == __name__
