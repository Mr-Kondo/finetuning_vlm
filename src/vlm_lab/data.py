"""Loading and ground-truth normalization for the CORD v2 dataset.

This module is the single source of truth for converting CORD v2's raw
Donut-format ``ground_truth`` JSON strings into a normalized structure.
Both training targets and evaluation references depend on
:func:`convert_ground_truth`, so correctness here matters more than
feature breadth.

**Held-out test blindness is a property of this module's API, not of caller
discipline.** The development loader below cannot name the test split, and
neither of the two loaders that can reach it lives here: Phase 5 uses
:mod:`vlm_lab.sealed_test` and the duplication audit uses
:mod:`vlm_lab.mechanical_access`. This module imports neither, so a Phase-2
module that stays within `vlm_lab.data` has no path to the test split at all
(review findings F5, R2-12, R3-14).
"""

from __future__ import annotations

import json

import datasets

CORD_V2_REPO_ID = "naver-clova-ix/cord-v2"

# The splits Phases 1-4 are permitted to read. Fixed in code so that no caller
# argument can extend the set.
DEVELOPMENT_SPLIT_NAMES = ("train", "validation")

# Every split. Only :mod:`vlm_lab.mechanical_access` may use this set.
ALL_SPLIT_NAMES = ("train", "validation", "test")

# gt_parse keys that Donut's serializer collapses from a one-element list to
# a bare dict. These must always be normalized back to a list.
_REPEATABLE_GROUP_KEYS = ("menu", "void_menu")
_REPEATABLE_NESTED_KEY = "sub"


def load_development_splits(revision: str | None = None) -> datasets.DatasetDict:
    """Load the CORD v2 train and validation splits, and nothing else.

    This is the loader for Phases 1-4, including ``notebooks/02_baseline.ipynb``.

    The capability is baked into the function: there is no parameter through
    which a caller can name a split, so the held-out test split is unreachable
    here whatever arguments are passed (R2-12). Each split is requested from the
    Hub by name, so `test` is never requested, downloaded, or materialized —
    filtering an already-loaded three-split `DatasetDict` would not have that
    property.

    ``revision`` is passed through to ``datasets.load_dataset`` untouched;
    ``None`` follows the Hub's default revision. Pinning it to the ADR-015
    commit SHA is a separate exit condition, not done here.
    """
    return load_named_splits(DEVELOPMENT_SPLIT_NAMES, revision)


def load_named_splits(
    split_names: tuple[str, ...],
    revision: str | None,
) -> datasets.DatasetDict:
    """Request each named split from the Hub individually and collect them.

    Shared low-level helper. It is *not* a capability boundary — the boundary is
    which module exposes which split set (see this module's docstring). Callers
    outside :mod:`vlm_lab.data`, :mod:`vlm_lab.sealed_test` and
    :mod:`vlm_lab.mechanical_access` should use one of those named entry points
    instead, so that every deliberate test-split read is logged.
    """
    return datasets.DatasetDict(
        {
            split_name: datasets.load_dataset(
                CORD_V2_REPO_ID, split=split_name, revision=revision
            )
            for split_name in split_names
        }
    )


def _as_list(value: object) -> list:
    """Wrap a bare dict in a single-element list; pass an existing list through."""
    if isinstance(value, dict):
        return [value]
    return value


def _require_dict_or_list_of_dicts(value: object, field_name: str) -> None:
    """Raise ValueError unless `value` is a dict, or a list whose elements are all dicts.

    This is the shape `_as_list` is designed to normalize. Anything else
    (a string, number, ``None``, or a list containing a non-dict element)
    is a structurally malformed `ground_truth` and must fail loudly rather
    than be silently coerced or misinterpreted (for example, iterating the
    characters of a string).
    """
    if isinstance(value, dict):
        return
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return
    raise ValueError(
        f"{field_name!r} must be a dict or a list of dicts, "
        f"got {type(value).__name__}: {value!r}"
    )


def _normalize_repeatable_group_item(item: dict) -> dict:
    """Normalize the nested `sub` key within a single menu/void_menu item."""
    if _REPEATABLE_NESTED_KEY not in item:
        return item
    sub = item[_REPEATABLE_NESTED_KEY]
    _require_dict_or_list_of_dicts(sub, _REPEATABLE_NESTED_KEY)
    normalized = dict(item)
    normalized[_REPEATABLE_NESTED_KEY] = _as_list(sub)
    return normalized


def convert_ground_truth(ground_truth_json: str) -> dict:
    """Convert a raw CORD v2 `ground_truth` JSON string into normalized `gt_parse`.

    Coerces the known repeatable-group keys (top-level `menu`, `void_menu`,
    and `sub` nested inside each of their items) to always be a list, since
    Donut's serialization collapses single-item groups to a bare dict. All
    other structure and values pass through unchanged.

    Raises:
        ValueError: if `ground_truth_json` is not valid JSON; if the parsed
            JSON is not an object; if it lacks a `gt_parse` key; if
            `gt_parse` is not an object; or if a repeatable-group key
            (`menu`, `void_menu`, or a nested `sub`) is present but is
            neither a dict nor a list of dicts.
    """
    try:
        parsed = json.loads(ground_truth_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ground_truth is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"ground_truth must decode to a JSON object, got {type(parsed).__name__}: {parsed!r}"
        )

    if "gt_parse" not in parsed:
        raise ValueError("ground_truth is missing required key 'gt_parse'")

    gt_parse = parsed["gt_parse"]
    if not isinstance(gt_parse, dict):
        raise ValueError(
            f"'gt_parse' must be a JSON object, got {type(gt_parse).__name__}: {gt_parse!r}"
        )

    normalized = dict(gt_parse)

    for key in _REPEATABLE_GROUP_KEYS:
        if key not in gt_parse:
            continue
        _require_dict_or_list_of_dicts(gt_parse[key], key)
        items = _as_list(gt_parse[key])
        normalized[key] = [_normalize_repeatable_group_item(item) for item in items]

    return normalized
