"""Unit tests for src/vlm_lab/schema.py's §6.5 schema-construction algorithm.

The corpora here are small and hand-written; nothing in this file touches
the Hub, and in particular nothing touches the held-out `test` split
(AGENTS.md §15, ADR-008). The load-bearing property is the round trip: a
schema generated from a corpus must accept that entire corpus. Review
finding R3-7 was exactly a case where it did not.
"""

from __future__ import annotations

import json

import pytest

from vlm_lab.schema import (
    JSON_SCHEMA_DRAFT_URI,
    SchemaShapeError,
    build_output_schema,
    schema_hash,
    validate_against_schema,
)


def reference_corpus() -> list[dict]:
    """A small CORD-shaped corpus: nested `sub`, an empty array, several leaves."""
    return [
        {
            "menu": [{"nm": "Coffee", "cnt": "1", "price": "10,000"}],
            "total": {"total_price": "10,000"},
        },
        {
            "menu": [
                {
                    "nm": "Combo Meal",
                    "price": "8,000",
                    "sub": [{"nm": "Fries", "price": "2,000"}],
                },
                {"nm": "Tea", "cnt": "2", "unitprice": "3,000", "price": "6,000"},
            ],
            "sub_total": {"subtotal_price": "14,000", "tax_price": "1,400"},
            "total": {"total_price": "15,400", "cashprice": "20,000", "changeprice": "4,600"},
            "void_menu": [],
        },
    ]


def test_root_is_a_strict_2020_12_object_with_no_required_keys():
    schema = build_output_schema(reference_corpus())

    assert schema["$schema"] == JSON_SCHEMA_DRAFT_URI
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == []


def test_property_set_is_exactly_the_observed_keys_at_each_level():
    schema = build_output_schema(reference_corpus())

    assert set(schema["properties"]) == {"menu", "sub_total", "total", "void_menu"}
    menu_item = schema["properties"]["menu"]["items"]
    assert set(menu_item["properties"]) == {"nm", "cnt", "price", "unitprice", "sub"}
    assert menu_item["additionalProperties"] is False
    assert menu_item["required"] == []
    assert menu_item["properties"]["nm"] == {"type": "string"}


def test_generated_schema_accepts_every_receipt_it_was_generated_from():
    corpus = reference_corpus()

    schema = build_output_schema(corpus)

    for receipt in corpus:
        assert validate_against_schema(receipt, schema) == []


def test_generated_schema_accepts_ground_truths_converted_by_data_module():
    # vlm_lab.data imports `datasets`. tests/conftest.py stubs it when absent, so
    # this normally runs; the guard only keeps the rest of this file collectable
    # in an environment that has neither.
    data = pytest.importorskip("vlm_lab.data")
    raw_ground_truths = [
        json.dumps({"gt_parse": {"menu": {"nm": "Coffee", "price": "10,000"}}}),
        json.dumps(
            {
                "gt_parse": {
                    "menu": [{"nm": "Combo", "sub": {"nm": "Fries", "price": "2,000"}}],
                    "total": {"total_price": "8,000"},
                }
            }
        ),
    ]
    corpus = [data.convert_ground_truth(raw) for raw in raw_ground_truths]

    schema = build_output_schema(corpus)

    for receipt in corpus:
        assert validate_against_schema(receipt, schema) == []


def test_generated_schema_accepts_a_minimal_single_menu_item_receipt():
    schema = build_output_schema(reference_corpus())

    assert validate_against_schema({"menu": [{"nm": "Coffee"}]}, schema) == []


def test_generated_schema_accepts_an_empty_array():
    schema = build_output_schema(reference_corpus())

    assert validate_against_schema({"menu": [], "void_menu": []}, schema) == []


def test_generated_schema_rejects_an_unknown_key():
    schema = build_output_schema(reference_corpus())

    violations = validate_against_schema({"menu": [], "grand_total": "10,000"}, schema)

    assert len(violations) == 1
    assert "grand_total" in violations[0]


def test_generated_schema_rejects_a_numeric_leaf():
    schema = build_output_schema(reference_corpus())

    violations = validate_against_schema({"menu": [{"nm": "Coffee", "price": 10000}]}, schema)

    assert violations == ["$.menu[0].price: 10000 is not of type 'string'"]


def test_generated_schema_rejects_a_null_leaf():
    schema = build_output_schema(reference_corpus())

    violations = validate_against_schema({"total": {"total_price": None}}, schema)

    assert violations == ["$.total.total_price: None is not of type 'string'"]


def test_generated_schema_rejects_a_bare_object_where_an_array_is_required():
    schema = build_output_schema(reference_corpus())

    violations = validate_against_schema({"menu": {"nm": "Coffee"}}, schema)

    assert len(violations) == 1
    assert violations[0].startswith("$.menu: ")
    assert "not of type 'array'" in violations[0]


def test_a_path_observed_only_as_empty_arrays_gets_no_item_constraint():
    corpus = [{"void_menu": []}, {"void_menu": []}]

    schema = build_output_schema(corpus)

    assert schema["properties"]["void_menu"] == {"type": "array"}


def test_object_mixed_with_array_at_one_path_raises_naming_the_path():
    corpus = [
        {"menu": [{"nm": "Coffee"}]},
        {"menu": {"nm": "Tea"}},
    ]

    with pytest.raises(SchemaShapeError) as raised:
        build_output_schema(corpus)

    assert "'$.menu'" in str(raised.value)
    assert "array" in str(raised.value)
    assert "object" in str(raised.value)


def test_string_mixed_with_object_at_one_path_raises_naming_the_path():
    corpus = [
        {"total": {"total_price": "10,000"}},
        {"total": "10,000"},
    ]

    with pytest.raises(SchemaShapeError) as raised:
        build_output_schema(corpus)

    assert "'$.total'" in str(raised.value)
    assert "object, string" in str(raised.value)


def test_array_of_scalars_raises_naming_the_element_path():
    corpus = [{"menu": ["Coffee", "Tea"]}]

    with pytest.raises(SchemaShapeError) as raised:
        build_output_schema(corpus)

    assert "'$.menu[]'" in str(raised.value)
    assert "string" in str(raised.value)


def test_non_string_scalar_leaf_raises_naming_the_path():
    corpus = [{"total": {"menuqty_cnt": 3}}]

    with pytest.raises(SchemaShapeError) as raised:
        build_output_schema(corpus)

    assert "'$.total.menuqty_cnt'" in str(raised.value)
    assert "integer" in str(raised.value)


def test_null_in_the_corpus_raises_rather_than_becoming_a_string_leaf():
    corpus = [{"total": {"total_price": "10,000"}}, {"total": {"total_price": None}}]

    with pytest.raises(SchemaShapeError) as raised:
        build_output_schema(corpus)

    assert "'$.total.total_price'" in str(raised.value)
    assert "null" in str(raised.value)


def test_nested_array_raises_naming_the_element_path():
    corpus = [{"menu": [[{"nm": "Coffee"}]]}]

    with pytest.raises(SchemaShapeError) as raised:
        build_output_schema(corpus)

    assert "'$.menu[]'" in str(raised.value)
    assert "array" in str(raised.value)


def test_empty_corpus_raises_rather_than_producing_an_empty_schema():
    with pytest.raises(ValueError, match="empty corpus"):
        build_output_schema([])


def test_non_object_corpus_element_raises_naming_its_index():
    with pytest.raises(ValueError, match="index 1"):
        build_output_schema([{"menu": []}, "not an object"])


def with_reversed_key_order(value: object) -> object:
    """Rebuild `value` with every dict's keys inserted in reverse order."""
    if isinstance(value, dict):
        return {key: with_reversed_key_order(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [with_reversed_key_order(item) for item in value]
    return value


def test_schema_hash_is_stable_across_dict_key_ordering():
    schema = build_output_schema(reference_corpus())
    reordered = with_reversed_key_order(schema)

    assert list(reordered) != list(schema)
    assert schema_hash(reordered) == schema_hash(schema)


def test_schema_hash_changes_when_the_schema_changes():
    schema = build_output_schema(reference_corpus())
    corpus_with_one_more_key = reference_corpus() + [{"total": {"creditcardprice": "9,000"}}]

    other_hash = schema_hash(build_output_schema(corpus_with_one_more_key))

    assert other_hash != schema_hash(schema)


def test_validate_against_schema_reports_every_violation_with_its_path():
    schema = build_output_schema(reference_corpus())

    violations = validate_against_schema(
        {"menu": [{"nm": 1}], "total": {"total_price": None}}, schema
    )

    assert violations == [
        "$.menu[0].nm: 1 is not of type 'string'",
        "$.total.total_price: None is not of type 'string'",
    ]
