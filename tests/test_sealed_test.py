"""Unit tests for src/vlm_lab/sealed_test.py and the access log it writes.

These tests patch the Hub boundary and never download anything: the point is
that the held-out test split is reachable only through this separate module,
and that reaching it leaves an audit record (review findings F5, R2-12, R3-14).
"""

from __future__ import annotations

import json

import pytest

from vlm_lab.access_log import DEFAULT_ACCESS_LOG_PATH, record_split_access
from vlm_lab.sealed_test import load_sealed_test_split


def test_load_sealed_test_split_requests_only_the_test_split(
    recorded_hub_split_requests, tmp_path
):
    load_sealed_test_split(access_log_path=tmp_path / "split_access.jsonl")

    assert [call["split"] for call in recorded_hub_split_requests] == ["test"]


def test_load_sealed_test_split_appends_an_access_log_entry(
    recorded_hub_split_requests, tmp_path
):
    access_log_path = tmp_path / "split_access.jsonl"

    load_sealed_test_split(access_log_path=access_log_path)

    entry = json.loads(access_log_path.read_text(encoding="utf-8").strip())
    assert entry["function"] == "load_sealed_test_split"
    assert entry["caller_module"] == __name__
    assert entry["timestamp_utc"].endswith("+00:00")
    # git_commit is None outside a checkout, so only its presence is asserted.
    assert "git_commit" in entry


def test_load_sealed_test_split_appends_rather_than_overwrites(
    recorded_hub_split_requests, tmp_path
):
    access_log_path = tmp_path / "split_access.jsonl"

    load_sealed_test_split(access_log_path=access_log_path)
    load_sealed_test_split(access_log_path=access_log_path)

    logged_lines = access_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(logged_lines) == 2


def test_load_sealed_test_split_creates_the_access_log_parent_directory(
    recorded_hub_split_requests, tmp_path
):
    access_log_path = tmp_path / "logs" / "split_access.jsonl"

    load_sealed_test_split(access_log_path=access_log_path)

    assert access_log_path.exists()


def test_default_access_log_path_lives_under_a_logs_directory():
    assert DEFAULT_ACCESS_LOG_PATH.parent.name == "logs"


def test_record_split_access_warns_but_does_not_raise_when_the_log_is_unwritable(
    tmp_path,
):
    # A logging failure must never abort the load it is recording, but it must
    # not pass unnoticed either.
    unwritable_path = tmp_path / "not-a-directory" / "split_access.jsonl"
    unwritable_path.parent.write_text("this is a file, not a directory", encoding="utf-8")

    with pytest.warns(UserWarning, match="split access log"):
        record_split_access("load_sealed_test_split", log_path=unwritable_path)
