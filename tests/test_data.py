"""Unit tests for src/vlm_lab/data.py's deterministic, network-independent logic.

These tests exercise `convert_ground_truth` against inline fixture JSON
strings, and the split-scoped loaders against a patched Hub boundary. They
never touch the network, so `pytest tests/` stays fast and can run without Hub
access. Network-dependent dataset loading is validated separately in the Phase 1
notebooks, per AGENTS.md's local-vs-Colab validation split.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import sys
from pathlib import Path

import pytest

import vlm_lab.data
from vlm_lab.data import (
    CORD_V2_REPO_ID,
    DEVELOPMENT_SPLIT_NAMES,
    convert_ground_truth,
    load_development_splits,
)


def test_convert_ground_truth_wraps_single_dict_menu_in_list():
    raw = json.dumps({"gt_parse": {"menu": {"nm": "Coffee", "price": "10,000"}}})

    result = convert_ground_truth(raw)

    assert result["menu"] == [{"nm": "Coffee", "price": "10,000"}]


def test_convert_ground_truth_preserves_multi_item_menu_and_wraps_nested_single_sub():
    raw = json.dumps(
        {
            "gt_parse": {
                "menu": [
                    {"nm": "Coffee", "price": "10,000"},
                    {
                        "nm": "Combo Meal",
                        "price": "8,000",
                        "sub": {"nm": "Lemon", "price": "1,000"},
                    },
                ]
            }
        }
    )

    result = convert_ground_truth(raw)

    assert len(result["menu"]) == 2
    assert result["menu"][0] == {"nm": "Coffee", "price": "10,000"}
    assert result["menu"][1]["sub"] == [{"nm": "Lemon", "price": "1,000"}]


def test_convert_ground_truth_preserves_existing_multi_item_sub_list_unchanged():
    raw = json.dumps(
        {
            "gt_parse": {
                "menu": [
                    {
                        "nm": "Combo Meal",
                        "sub": [
                            {"nm": "Fries", "price": "2,000"},
                            {"nm": "Drink", "price": "1,500"},
                        ],
                    }
                ]
            }
        }
    )

    result = convert_ground_truth(raw)

    assert result["menu"][0]["sub"] == [
        {"nm": "Fries", "price": "2,000"},
        {"nm": "Drink", "price": "1,500"},
    ]


def test_convert_ground_truth_does_not_fabricate_missing_sub_total_key():
    raw = json.dumps(
        {
            "gt_parse": {
                "menu": {"nm": "Coffee", "price": "10,000"},
                "total": {"total_price": "10,000"},
            }
        }
    )

    result = convert_ground_truth(raw)

    assert "sub_total" not in result


def test_convert_ground_truth_wraps_single_dict_void_menu_in_list():
    raw = json.dumps({"gt_parse": {"void_menu": {"nm": "Cancelled Item", "price": "5,000"}}})

    result = convert_ground_truth(raw)

    assert result["void_menu"] == [{"nm": "Cancelled Item", "price": "5,000"}]


def test_convert_ground_truth_raises_value_error_on_malformed_json():
    malformed = "{gt_parse: not valid json"

    with pytest.raises(ValueError) as exc_info:
        convert_ground_truth(malformed)

    assert str(exc_info.value)


def test_convert_ground_truth_raises_value_error_when_gt_parse_key_missing():
    raw = json.dumps({"unrelated_key": {"menu": {"nm": "Coffee"}}})

    with pytest.raises(ValueError) as exc_info:
        convert_ground_truth(raw)

    assert str(exc_info.value)


def test_convert_ground_truth_does_not_normalize_string_or_number_values():
    raw = json.dumps(
        {
            "gt_parse": {
                "menu": {"nm": "Coffee", "price": "40,000."},
                "total": {
                    "total_price": "40,000.",
                    "cashprice": "50,000",
                    "changeprice": "10,000",
                },
            }
        }
    )

    result = convert_ground_truth(raw)

    # The trailing period and comma-formatted digits must pass through
    # verbatim: convert_ground_truth performs structural normalization only,
    # never string/number normalization.
    assert result["menu"] == [{"nm": "Coffee", "price": "40,000."}]
    assert result["total"] == {
        "total_price": "40,000.",
        "cashprice": "50,000",
        "changeprice": "10,000",
    }


def test_convert_ground_truth_raises_value_error_when_top_level_json_is_not_an_object():
    raw = json.dumps(["gt_parse", {"menu": {"nm": "Coffee"}}])

    with pytest.raises(ValueError) as exc_info:
        convert_ground_truth(raw)

    assert str(exc_info.value)


def test_convert_ground_truth_raises_value_error_when_top_level_json_is_null():
    raw = json.dumps(None)

    with pytest.raises(ValueError) as exc_info:
        convert_ground_truth(raw)

    assert str(exc_info.value)


def test_convert_ground_truth_raises_value_error_when_gt_parse_is_a_list():
    raw = json.dumps({"gt_parse": []})

    with pytest.raises(ValueError) as exc_info:
        convert_ground_truth(raw)

    assert str(exc_info.value)


def test_convert_ground_truth_raises_value_error_when_gt_parse_is_null():
    raw = json.dumps({"gt_parse": None})

    with pytest.raises(ValueError) as exc_info:
        convert_ground_truth(raw)

    assert str(exc_info.value)


def test_convert_ground_truth_raises_value_error_when_menu_is_a_string():
    raw = json.dumps({"gt_parse": {"menu": "abc"}})

    with pytest.raises(ValueError) as exc_info:
        convert_ground_truth(raw)

    assert str(exc_info.value)


def test_convert_ground_truth_raises_value_error_when_menu_is_a_list_of_non_dicts():
    raw = json.dumps({"gt_parse": {"menu": ["nm", "price"]}})

    with pytest.raises(ValueError) as exc_info:
        convert_ground_truth(raw)

    assert str(exc_info.value)


def test_convert_ground_truth_raises_value_error_when_void_menu_is_a_number():
    raw = json.dumps({"gt_parse": {"void_menu": 42}})

    with pytest.raises(ValueError) as exc_info:
        convert_ground_truth(raw)

    assert str(exc_info.value)


def test_convert_ground_truth_raises_value_error_when_nested_sub_is_a_string():
    raw = json.dumps(
        {"gt_parse": {"menu": {"nm": "Combo Meal", "sub": "not-a-dict-or-list"}}}
    )

    with pytest.raises(ValueError) as exc_info:
        convert_ground_truth(raw)

    assert str(exc_info.value)


def test_convert_ground_truth_raises_value_error_when_nested_sub_list_has_a_non_dict_element():
    raw = json.dumps(
        {
            "gt_parse": {
                "menu": {
                    "nm": "Combo Meal",
                    "sub": [{"nm": "Fries", "price": "2,000"}, "not-a-dict"],
                }
            }
        }
    )

    with pytest.raises(ValueError) as exc_info:
        convert_ground_truth(raw)

    assert str(exc_info.value)


def test_cord_v2_repo_id_constant_matches_expected_hub_repo():
    assert CORD_V2_REPO_ID == "naver-clova-ix/cord-v2"


def test_load_development_splits_never_requests_the_test_split_from_the_hub(
    recorded_hub_split_requests,
):
    # Asserting on the *request* rather than on the returned object is the point
    # (R2-12): a loader that downloaded all three splits and then dropped `test`
    # would still return a test-free DatasetDict, and would still have read it.
    load_development_splits()

    requested_splits = [call["split"] for call in recorded_hub_split_requests]
    assert requested_splits == list(DEVELOPMENT_SPLIT_NAMES)
    assert "test" not in requested_splits
    assert all(call["repo_id"] == CORD_V2_REPO_ID for call in recorded_hub_split_requests)


def test_load_development_splits_passes_the_revision_through_to_the_hub(
    recorded_hub_split_requests,
):
    load_development_splits(revision="7f0115a4b758a71d6473b8d085751692da2fef98")

    assert all(
        call["revision"] == "7f0115a4b758a71d6473b8d085751692da2fef98"
        for call in recorded_hub_split_requests
    )


def test_load_development_splits_has_no_parameter_that_can_name_a_split():
    # The capability lives in the function, not in an argument: `revision` is the
    # only knob, so there is nothing a caller can set to "test" (F5, R2-12).
    parameter_names = list(inspect.signature(load_development_splits).parameters)

    assert parameter_names == ["revision"]


def test_load_development_splits_returns_only_train_and_validation(
    recorded_hub_split_requests,
):
    result = load_development_splits()

    assert set(result.keys()) == {"train", "validation"}


def test_data_module_does_not_import_the_sealed_test_module():
    # R3-14: Phase-2 code that stays inside `vlm_lab.data` must have no import
    # path to the sealed loader at all. Checked both statically (no import
    # statement) and dynamically (a fresh import does not pull it in).
    imported_modules = _module_names_imported_by(Path(vlm_lab.data.__file__))
    assert not any(name.endswith("sealed_test") for name in imported_modules)

    preserved_modules = {
        name: module for name, module in sys.modules.items() if name.startswith("vlm_lab")
    }
    try:
        for name in preserved_modules:
            del sys.modules[name]
        importlib.import_module("vlm_lab.data")

        assert "vlm_lab.sealed_test" not in sys.modules
    finally:
        sys.modules.update(preserved_modules)


@pytest.mark.skip(
    reason="network: downloads the real naver-clova-ix/cord-v2 dataset from "
    "the Hugging Face Hub; excluded from the default fast local suite. "
    "Run manually (remove this marker) when Hub access and time budget "
    "allow, or exercise it from the Phase 1 notebook instead."
)
def test_load_development_splits_downloads_train_and_validation_from_the_real_hub():
    result = load_development_splits()

    assert set(result.keys()) == {"train", "validation"}


def _module_names_imported_by(module_path: Path) -> list[str]:
    """Return every module name imported by `module_path`, from its source AST."""
    imported_names: list[str] = []
    for node in ast.walk(ast.parse(module_path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_names.append(node.module)
            imported_names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return imported_names
