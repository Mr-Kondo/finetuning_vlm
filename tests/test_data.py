"""Unit tests for src/vlm_lab/data.py's deterministic, network-independent logic.

These tests exercise `convert_ground_truth` against inline fixture JSON
strings only. They do not call `load_cord_v2` or otherwise touch the
network, so `pytest tests/` stays fast and can run without Hub access.
Network-dependent dataset loading is validated separately in the Phase 1
notebooks, per AGENTS.md's local-vs-Colab validation split.
"""

from __future__ import annotations

import json

import pytest

from vlm_lab.data import CORD_V2_REPO_ID, convert_ground_truth, load_cord_v2


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


@pytest.mark.skip(
    reason="network: downloads the real naver-clova-ix/cord-v2 dataset from "
    "the Hugging Face Hub; excluded from the default fast local suite. "
    "Run manually (remove this marker) when Hub access and time budget "
    "allow, or exercise it from the Phase 1 notebook instead."
)
def test_load_cord_v2_downloads_dataset_dict_with_expected_splits():
    result = load_cord_v2()

    assert set(result.keys()) == {"train", "validation", "test"}
