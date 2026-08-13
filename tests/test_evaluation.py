"""Tests for the shared evaluation path.

The first section is the P-11b synthetic-error probe frozen in
``docs/proposals/phase1_closure_prereg.md`` §6.1: six fixed predictions against
one fixed reference, with both metrics' values asserted exactly. It is a
*characterization* table, not a pass/fail quality bar — its job is to record how
each metric responds to each error class before any model output exists, so that
a later change to the evaluator or to ``zss`` / ``nltk`` shows up as a fixture
diff rather than as a silent metric shift.
"""

from __future__ import annotations

import json
import logging

import pytest

from vlm_lab.evaluation import (
    ParseOutcome,
    field_value_multiset_f1,
    flatten_field_values,
    parse_raw_output,
    raw_json_validity_rate,
    recoverable_payload_rate,
    ted_accuracy,
)
from vlm_lab.third_party.donut_eval import JSONParseEvaluator

# The frozen reference receipt shared by all six probe fixtures (§6.1).
PROBE_REFERENCE = {
    "menu": [
        {"nm": "SPGTHY BOLOGNASE", "cnt": "1", "price": "58,000"},
        {"nm": "ICED LEMON TEA", "cnt": "1", "price": "22,000"},
    ],
    "total": {"total_price": "80,000"},
}

# Fixture 5 is defined by §6.1 as "the reference serialization with its final
# `}` removed", which is what makes it unparseable.
TRUNCATED_REFERENCE_OUTPUT = json.dumps(PROBE_REFERENCE)[:-1]

# (case, raw model output, expected TED-Acc, expected index-free field-value multiset F1)
PROBE_FIXTURES = [
    pytest.param(
        json.dumps(
            {
                "menu": [
                    {"nm": "SPGTHY BOLOGNASE", "cnt": "1", "price": "59,000"},
                    {"nm": "ICED LEMON TEA", "cnt": "1", "price": "22,000"},
                ],
                "total": {"total_price": "80,000"},
            }
        ),
        0.9838709677419355,
        0.8571428571428571,
        id="1-wrong-amount-digit",
    ),
    pytest.param(
        json.dumps(
            {
                "menu": [
                    {"nm": "SPGTHY BOLOGNASE", "price": "58,000"},
                    {"nm": "ICED LEMON TEA", "cnt": "1", "price": "22,000"},
                ],
                "total": {"total_price": "80,000"},
            }
        ),
        0.967741935483871,
        0.9230769230769231,
        id="2-missing-field",
    ),
    pytest.param(
        json.dumps(
            {
                "menu": [
                    {
                        "nm": "SPGTHY BOLOGNASE",
                        "cnt": "1",
                        "price": "58,000",
                        "unitprice": "58,000",
                    },
                    {"nm": "ICED LEMON TEA", "cnt": "1", "price": "22,000"},
                ],
                "total": {"total_price": "80,000"},
            }
        ),
        0.8870967741935484,
        0.9333333333333333,
        id="3-extra-field",
    ),
    pytest.param(
        json.dumps(
            {
                "menu": [
                    {"nm": "ICED LEMON TEA", "cnt": "1", "price": "22,000"},
                    {"nm": "SPGTHY BOLOGNASE", "cnt": "1", "price": "58,000"},
                ],
                "total": {"total_price": "80,000"},
            }
        ),
        0.5161290322580645,
        1.0,
        id="4-item-reorder",
    ),
    pytest.param(
        TRUNCATED_REFERENCE_OUTPUT,
        0.0,
        0.0,
        id="5-invalid-json",
    ),
    pytest.param(
        json.dumps(
            {
                "menu": [
                    {"nm": "SPGHTY BOLOGNASE", "cnt": "1", "price": "58,000"},
                    {"nm": "ICED LEMON TEA", "cnt": "1", "price": "22,000"},
                ],
                "total": {"total_price": "80,000"},
            }
        ),
        0.967741935483871,
        0.8571428571428571,
        id="6-near-correct-ocr",
    ),
]


@pytest.mark.parametrize(
    "raw_output, expected_ted_accuracy, expected_field_value_f1", PROBE_FIXTURES
)
def test_synthetic_error_probe_characterizes_both_metrics(
    raw_output: str, expected_ted_accuracy: float, expected_field_value_f1: float
) -> None:
    """P-11b: each frozen fixture scores exactly these values (§6.1)."""
    payload = parse_raw_output(raw_output).payload

    assert ted_accuracy(payload, PROBE_REFERENCE) == pytest.approx(expected_ted_accuracy)
    assert field_value_multiset_f1(payload, PROBE_REFERENCE) == pytest.approx(
        expected_field_value_f1
    )


def test_ted_accuracy_is_nearly_blind_to_a_wrong_amount_digit() -> None:
    """§6.1's defined consequence: TED-Acc >= 0.95 on fixture 1 must be visible.

    `58,000` -> `59,000` is a materially wrong amount but costs one character
    edit out of a 62-cost reference tree. This assertion exists so the fact
    cannot quietly disappear if the fixture values are ever regenerated: it is
    the trigger for the `USER DECISION REQUIRED` escalation in §6.1, and it is
    why the field-value multiset diagnostic is mandatory rather than optional.
    """
    wrong_amount_prediction = {
        "menu": [
            {"nm": "SPGTHY BOLOGNASE", "cnt": "1", "price": "59,000"},
            {"nm": "ICED LEMON TEA", "cnt": "1", "price": "22,000"},
        ],
        "total": {"total_price": "80,000"},
    }

    assert ted_accuracy(wrong_amount_prediction, PROBE_REFERENCE) >= 0.95
    assert field_value_multiset_f1(wrong_amount_prediction, PROBE_REFERENCE) < 0.95


def test_ted_accuracy_penalizes_item_reorder_that_the_diagnostic_ignores() -> None:
    """The two metrics disagree hardest on row order, in opposite directions.

    ``zss`` computes an *ordered* tree edit distance, so swapping two menu rows
    whose values are all correct costs 30 of the reference tree's 62 units. The
    index-free multiset is invariant to the same change by construction. Neither
    behaviour is a defect; both must be visible when the pair is reported.
    """
    reordered_prediction = {
        "menu": [
            {"nm": "ICED LEMON TEA", "cnt": "1", "price": "22,000"},
            {"nm": "SPGTHY BOLOGNASE", "cnt": "1", "price": "58,000"},
        ],
        "total": {"total_price": "80,000"},
    }

    assert ted_accuracy(reordered_prediction, PROBE_REFERENCE) == pytest.approx(
        0.5161290322580645
    )
    assert field_value_multiset_f1(reordered_prediction, PROBE_REFERENCE) == 1.0


class TestParsing:
    """§6.3's raw-validity, fence-recovery and parse-failure rules."""

    def test_unfenced_json_object_is_raw_valid(self) -> None:
        outcome = parse_raw_output('{"total": {"total_price": "80,000"}}')

        assert outcome == ParseOutcome(
            raw_json_is_valid=True,
            recovered_from_fence=False,
            payload={"total": {"total_price": "80,000"}},
        )

    def test_fenced_output_is_not_raw_valid_but_is_recoverable(self) -> None:
        outcome = parse_raw_output('```json\n{"total": {"total_price": "80,000"}}\n```')

        assert not outcome.raw_json_is_valid
        assert outcome.recovered_from_fence
        assert outcome.payload == {"total": {"total_price": "80,000"}}

    def test_fence_without_an_info_string_is_recoverable(self) -> None:
        outcome = parse_raw_output('```\n{"a": "1"}\n```')

        assert not outcome.raw_json_is_valid
        assert outcome.payload == {"a": "1"}

    def test_the_two_parse_rates_differ_when_some_outputs_are_fenced(self) -> None:
        """Both rates are published, so a fencing habit is visible not absorbed."""
        outcomes = [
            parse_raw_output('{"a": "1"}'),
            parse_raw_output('```json\n{"a": "1"}\n```'),
            parse_raw_output('{"a": "1"'),
            parse_raw_output('{"a": "1"}'),
        ]

        assert raw_json_validity_rate(outcomes) == 0.5
        assert recoverable_payload_rate(outcomes) == 0.75

    def test_prose_around_a_fenced_block_is_not_recoverable(self) -> None:
        """Recovery applies only when the *entire* output is one fence (§6.3)."""
        outcome = parse_raw_output('Here you go:\n```json\n{"a": "1"}\n```\nHope that helps!')

        assert not outcome.raw_json_is_valid
        assert not outcome.recovered_from_fence
        assert outcome.payload is None

    def test_two_fenced_blocks_are_not_recoverable(self) -> None:
        outcome = parse_raw_output('```json\n{"a": "1"}\n```\n```json\n{"b": "2"}\n```')

        assert outcome.payload is None

    def test_non_object_top_level_is_a_parse_failure(self) -> None:
        for raw_output in ('[{"a": "1"}]', '"just a string"', "42", "null"):
            outcome = parse_raw_output(raw_output)

            assert not outcome.raw_json_is_valid, raw_output
            assert outcome.payload is None, raw_output

    def test_empty_object_is_raw_valid(self) -> None:
        """`{}` is well-formed output that happens to say nothing (§6.3)."""
        outcome = parse_raw_output("{}")

        assert outcome.raw_json_is_valid
        assert outcome.payload == {}

    def test_parse_failure_scores_zero_on_every_content_metric(self) -> None:
        """ADR-011: scored 0, never dropped from the denominator."""
        payload = parse_raw_output("not json at all").payload

        assert payload is None
        assert ted_accuracy(payload, PROBE_REFERENCE) == 0.0
        assert field_value_multiset_f1(payload, PROBE_REFERENCE) == 0.0


class TestTrimmedVerbatimNormalization:
    """ADR-022: strip leading/trailing whitespace, compare everything else exactly."""

    def test_digit_grouping_is_not_normalized(self) -> None:
        assert field_value_multiset_f1({"price": "58000"}, {"price": "58,000"}) == 0.0

    def test_leading_and_trailing_whitespace_is_ignored(self) -> None:
        assert field_value_multiset_f1({"price": "  58,000\n"}, {"price": "58,000"}) == 1.0

    def test_internal_whitespace_is_not_collapsed(self) -> None:
        assert field_value_multiset_f1({"nm": "ICED  LEMON TEA"}, {"nm": "ICED LEMON TEA"}) == 0.0

    def test_case_is_not_folded(self) -> None:
        assert field_value_multiset_f1({"nm": "iced lemon tea"}, {"nm": "ICED LEMON TEA"}) == 0.0

    def test_full_width_digits_are_not_normalized(self) -> None:
        """No NFKC: full-width `５８` stays distinct from `58`."""
        assert field_value_multiset_f1({"price": "５８,000"}, {"price": "58,000"}) == 0.0

    def test_a_value_that_is_empty_after_trimming_counts_as_absent(self) -> None:
        assert flatten_field_values({"nm": "A", "cnt": "   "}) == flatten_field_values({"nm": "A"})
        assert field_value_multiset_f1({"nm": "A", "cnt": "   "}, {"nm": "A"}) == 1.0

    def test_whitespace_only_leaves_are_dropped_by_only_one_of_the_metrics(self) -> None:
        """The two metrics apply §6.3's empty-value rule differently. Characterized here.

        `field_value_multiset_f1` implements §6.1 step 2 literally — trim, then
        drop what is empty. The pinned `normalize_dict` checks falsiness *before*
        stripping, so a whitespace-only leaf survives into the tree and costs
        TED-Acc the key node. Making them agree would require patching the
        vendored evaluator, which ADR-022 explicitly declined to do.
        """
        prediction = {"a": "x", "b": "   "}
        reference = {"a": "x"}

        assert field_value_multiset_f1(prediction, reference) == 1.0
        assert ted_accuracy(prediction, reference) == pytest.approx(0.5)

    def test_null_and_absent_are_identical_on_both_sides(self) -> None:
        assert field_value_multiset_f1({"nm": "A", "cnt": None}, {"nm": "A"}) == 1.0
        assert field_value_multiset_f1({"nm": "A"}, {"nm": "A", "cnt": None}) == 1.0


class TestFieldValueMultisetF1:
    """§6.1 P-11a, including the limitation ADR-022 requires to be stated."""

    def test_array_indices_are_removed_from_the_path(self) -> None:
        flattened = flatten_field_values({"menu": [{"nm": "A"}, {"nm": "B"}]})

        assert flattened == {("menu[].nm", "A"): 1, ("menu[].nm", "B"): 1}

    def test_duplicate_rows_must_be_duplicated_correctly_to_earn_credit(self) -> None:
        """A multiset, not a set: multiplicity is counted."""
        two_identical_rows = {"menu": [{"nm": "A"}, {"nm": "A"}]}
        one_row = {"menu": [{"nm": "A"}]}

        assert flatten_field_values(two_identical_rows) == {("menu[].nm", "A"): 2}
        assert field_value_multiset_f1(one_row, two_identical_rows) == pytest.approx(2 / 3)

    def test_it_is_blind_to_row_association(self) -> None:
        """ADR-022's documented limitation, asserted so it cannot be forgotten.

        Index-free paths make this a *bag* of field values, so a prediction that
        pairs each name with the other row's price still scores 1.0. This is why
        the metric is named a multiset F1 and designated a diagnostic rather than
        a field-exact guardrail.
        """
        reference = {"menu": [{"nm": "A", "price": "1"}, {"nm": "B", "price": "2"}]}
        prices_swapped = {"menu": [{"nm": "A", "price": "2"}, {"nm": "B", "price": "1"}]}

        assert field_value_multiset_f1(prices_swapped, reference) == 1.0

    def test_two_empty_documents_score_one(self) -> None:
        assert field_value_multiset_f1({}, {}) == 1.0

    def test_exactly_one_empty_document_scores_zero(self) -> None:
        assert field_value_multiset_f1({}, PROBE_REFERENCE) == 0.0
        assert field_value_multiset_f1(PROBE_REFERENCE, {}) == 0.0

    def test_a_numeric_leaf_matches_its_string_counterpart(self) -> None:
        """§6.2 consequence 4: content credit and schema conformance diverge.

        The vendored evaluator stringifies every leaf, so the diagnostic does
        too. Type conformance is reported separately as strict schema validity.
        """
        assert field_value_multiset_f1({"cnt": 1}, {"cnt": "1"}) == 1.0


class TestEmptyReference:
    """§6.2: `cal_acc` divides by `TED(empty, answer)`, which is 0 for an empty answer."""

    def test_the_pinned_evaluator_really_does_raise_zero_division(self) -> None:
        """The hazard the wrapper handles is real, not hypothetical."""
        with pytest.raises(ZeroDivisionError):
            JSONParseEvaluator().cal_acc({}, {})

    def test_empty_reference_scores_one_for_an_empty_prediction(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            score = ted_accuracy({}, {})

        assert score == 1.0
        assert "empty reference" in caplog.text

    def test_empty_reference_scores_zero_for_a_non_empty_prediction(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            score = ted_accuracy({"total": {"total_price": "80,000"}}, {})

        assert score == 0.0
        assert "empty reference" in caplog.text

    def test_a_reference_whose_every_value_is_an_empty_string_is_an_empty_reference(
        self,
    ) -> None:
        """`normalize_dict` drops falsy leaves, so this reference normalizes away."""
        assert ted_accuracy({}, {"total": {"total_price": ""}}) == 1.0

    def test_a_whitespace_only_reference_value_is_not_an_empty_reference(self) -> None:
        """Characterizes a divergence between §6.3's wording and the pinned evaluator.

        §6.3 says a leaf that is empty *after trimming* counts as absent "matching
        the vendored evaluator's `normalize_dict`". That is only true for values
        that are already falsy: `normalize_dict` tests `if not data` *before*
        stripping, so `"   "` survives as `[""]` — a `<leaf>` with an empty label
        whose parent key node still costs 1 to insert. TED-Acc therefore
        penalizes a whitespace-only leaf that the field-value multiset drops.
        See `test_whitespace_only_leaves_are_dropped_by_only_one_of_the_metrics`.
        """
        assert ted_accuracy({}, {"total": {"total_price": "   "}}) == 0.0


class TestRegressionFixtures:
    """§6.2 requirement 4: known scores that pin `zss` / `nltk` behaviour.

    These are deliberately small enough to verify by hand. The reference tree of
    `{"a": "abc"}` costs 1 (the `a` node) + 3 (the leaf label `abc`) = 4 to
    insert from empty, which is the denominator every score below divides by.
    """

    def test_identical_documents_score_one_on_both_metrics(self) -> None:
        assert ted_accuracy(PROBE_REFERENCE, PROBE_REFERENCE) == 1.0
        assert field_value_multiset_f1(PROBE_REFERENCE, PROBE_REFERENCE) == 1.0

    def test_empty_prediction_against_a_real_reference_scores_zero(self) -> None:
        assert ted_accuracy({}, PROBE_REFERENCE) == 0.0

    def test_one_character_substitution_costs_one_of_four(self) -> None:
        assert ted_accuracy({"a": "abd"}, {"a": "abc"}) == pytest.approx(0.75)

    def test_one_character_deletion_costs_one_of_four(self) -> None:
        assert ted_accuracy({"a": "ab"}, {"a": "abc"}) == pytest.approx(0.75)

    def test_a_wrong_key_costs_one_relabel_of_four(self) -> None:
        """`zss` relabels the `a` node to `b` rather than deleting and reinserting it.

        A wrong key therefore costs the same as a one-character value typo here,
        which is exactly the kind of behaviour a frozen fixture is for.
        """
        assert ted_accuracy({"b": "abc"}, {"a": "abc"}) == pytest.approx(0.75)

    def test_ted_accuracy_is_floored_at_zero(self) -> None:
        """`max(0, 1 - nTED)`: a prediction far larger than the reference cannot go negative."""
        oversized_prediction = {f"k{index}": "x" * 50 for index in range(20)}

        assert ted_accuracy(oversized_prediction, {"a": "abc"}) == 0.0
