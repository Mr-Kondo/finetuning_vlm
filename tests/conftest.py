"""Test-suite bootstrap.

`vlm_lab.data` and `vlm_lab.sealed_test` import `datasets` at module level, but
`datasets` is a heavy, network- and GPU-adjacent dependency that the fast local
suite should not need (AGENTS.md §25's local-vs-Colab validation split). When it
is genuinely installed, the tests use it. When it is not, this module installs a
minimal stand-in so the Hub boundary is still patchable and the split-request
assertions still run.

The stand-in's `load_dataset` raises if it is ever reached, so a test that
forgets to patch the boundary fails loudly instead of passing vacuously.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest


def _install_datasets_stub_if_missing() -> None:
    try:
        import datasets  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["datasets"] = _minimal_datasets_stub()


def _minimal_datasets_stub() -> ModuleType:
    stub = ModuleType("datasets")

    class Dataset:
        """Stand-in for `datasets.Dataset`; tests only pass these around."""

    class DatasetDict(dict):
        """Stand-in for `datasets.DatasetDict`, which is a dict subclass upstream."""

    def load_dataset(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "datasets.load_dataset was called for real. Tests must patch this "
            "Hub boundary; the real loader is unavailable in the fast local suite."
        )

    stub.Dataset = Dataset
    stub.DatasetDict = DatasetDict
    stub.load_dataset = load_dataset
    return stub


_install_datasets_stub_if_missing()


@pytest.fixture
def recorded_hub_split_requests(monkeypatch):
    """Patch `datasets.load_dataset` and return the list of requests made to it.

    Each recorded request is a dict with `repo_id`, `split` and `revision`. The
    loaders under test look `load_dataset` up on the `datasets` module at call
    time, so patching the attribute there intercepts the real Hub boundary.
    """
    import datasets

    recorded_requests: list[dict[str, object]] = []

    def record_request(repo_id, split=None, revision=None, **kwargs):
        recorded_requests.append(
            {"repo_id": repo_id, "split": split, "revision": revision, **kwargs}
        )
        return datasets.Dataset()

    monkeypatch.setattr(datasets, "load_dataset", record_request)
    return recorded_requests
