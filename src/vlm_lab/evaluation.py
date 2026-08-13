"""Shared evaluation path for the Base and Fine-tuned conditions.

ADR-006 requires both conditions to be scored by the same code. Nothing in this
module takes a condition argument, so there is no place for a rule that could
favour one model over the other.

The rules implemented here are frozen in
``docs/proposals/phase1_closure_prereg.md`` §6.1-§6.3:

* **Raw JSON validity** and **recoverable payload** are two different facts and
  are reported as two rates (§6.3). Collapsing them would hide a Base /
  Fine-tuned difference in markdown-fencing habits inside a single number.
* The estimand is **trimmed verbatim transcription** (ADR-022): leading and
  trailing whitespace is ignored, everything else — internal spacing, digit
  grouping, currency symbols, case, full/half-width forms — is compared exactly.
* **TED-Acc** is the primary metric, computed by the pinned Donut evaluator
  vendored in :mod:`vlm_lab.third_party.donut_eval`.
* **Index-free field-value multiset F1** is a mandatory *diagnostic* (ADR-022),
  never a gate. Its documented limitation must be stated wherever it is
  reported: replacing array indices with ``[]`` makes it a bag of field values,
  so it is blind to row association (reference rows ``(A, 1)`` and ``(B, 2)``
  score 1.0 against a prediction of ``(A, 2)`` and ``(B, 1)``).
* A parse failure scores 0 on every content metric and is never dropped from the
  denominator (ADR-011).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from vlm_lab.third_party.donut_eval import JSONParseEvaluator

logger = logging.getLogger(__name__)

MARKDOWN_FENCE = "```"

_donut_evaluator = JSONParseEvaluator()

# A multiset of (index-free dotted path, trimmed value) pairs.
FieldValueCounts = Counter[tuple[str, str]]


@dataclass(frozen=True)
class ParseOutcome:
    """What one raw model output yielded, per §6.3's first three rules.

    Attributes:
        raw_json_is_valid: the *unmodified* output parses as a JSON object.
            Fenced output is not raw-valid, because the prompt forbids fences.
        recovered_from_fence: the output was not raw-valid, but the whole output
            was wrapped in exactly one markdown fence and the fenced text parsed.
        payload: the object that content metrics are computed from, or ``None``
            for a parse failure.
    """

    raw_json_is_valid: bool
    recovered_from_fence: bool
    payload: dict | None

    @property
    def has_payload(self) -> bool:
        """Whether any payload was obtained, raw or recovered."""
        return self.payload is not None


def parse_raw_output(text: str) -> ParseOutcome:
    """Parse one raw model output into a :class:`ParseOutcome` (§6.3).

    Raw validity is judged on `text` exactly as the model emitted it. Only when
    that fails is fence recovery attempted, and only when the *entire* output is
    wrapped in exactly one markdown fence. A top level that is not a JSON object
    (a list, a bare string, a number) is a parse failure.
    """
    payload = _loads_json_object(text)
    if payload is not None:
        return ParseOutcome(raw_json_is_valid=True, recovered_from_fence=False, payload=payload)

    fenced_text = _unwrap_single_markdown_fence(text)
    if fenced_text is not None:
        payload = _loads_json_object(fenced_text)
        if payload is not None:
            return ParseOutcome(
                raw_json_is_valid=False, recovered_from_fence=True, payload=payload
            )

    return ParseOutcome(raw_json_is_valid=False, recovered_from_fence=False, payload=None)


def raw_json_validity_rate(outcomes: Sequence[ParseOutcome]) -> float:
    """Fraction of outputs that parsed as a JSON object unmodified (§6.1)."""
    return _fraction(outcome.raw_json_is_valid for outcome in outcomes)


def recoverable_payload_rate(outcomes: Sequence[ParseOutcome]) -> float:
    """Fraction of outputs that yielded a payload, raw or fence-recovered (§6.1).

    Reported next to :func:`raw_json_validity_rate` rather than instead of it:
    the gap between the two rates *is* the fencing habit, and content metrics
    are computed from the recovered payload.
    """
    return _fraction(outcome.has_payload for outcome in outcomes)


def ted_accuracy(prediction: dict | None, reference: dict) -> float:
    """Score one receipt with Donut's normalized tree edit distance accuracy.

    This is the primary metric (ADR-020). `prediction` is ``None`` for a parse
    failure, which scores 0.0.

    The pinned ``cal_acc`` divides by ``TED(empty_tree, reference_tree)``, which
    is zero for a reference that normalizes to empty and would raise
    ``ZeroDivisionError``. Per §6.2 that case is handled here instead of
    crashing: an empty reference scores 1.0 if the prediction is also empty and
    0.0 otherwise, and the occurrence is logged. No CORD receipt is empty, so
    this firing at all is itself a finding.
    """
    if prediction is None:
        return 0.0

    if not _donut_evaluator.normalize_dict(reference):
        prediction_is_empty = not _donut_evaluator.normalize_dict(prediction)
        logger.warning(
            "empty reference: cal_acc would divide by TED(empty, empty) == 0. "
            "Scoring %.1f (the prediction is %s). Reference: %r",
            1.0 if prediction_is_empty else 0.0,
            "also empty" if prediction_is_empty else "not empty",
            reference,
        )
        return 1.0 if prediction_is_empty else 0.0

    return _donut_evaluator.cal_acc(prediction, reference)


def field_value_multiset_f1(prediction: dict | None, reference: dict) -> float:
    """Score one receipt with the index-free field-value multiset F1 (§6.1 P-11a).

    A mandatory diagnostic, never a gate (ADR-022). `prediction` is ``None`` for
    a parse failure, which scores 0.0. Both sides are flattened to a multiset of
    ``(path, value)`` pairs with array indices removed from the path, so the
    score is invariant to menu-row ordering while still counting multiplicity.

    Wherever this is reported, report its limitation with it: index-free paths
    make it a *bag* of field values, so it is blind to row and parent-child
    association (ADR-022).
    """
    if prediction is None:
        return 0.0

    predicted_fields = flatten_field_values(prediction)
    reference_fields = flatten_field_values(reference)
    if not predicted_fields and not reference_fields:
        return 1.0

    true_positives = sum((predicted_fields & reference_fields).values())
    false_positives = sum(predicted_fields.values()) - true_positives
    false_negatives = sum(reference_fields.values()) - true_positives
    return 2 * true_positives / (2 * true_positives + false_positives + false_negatives)


def flatten_field_values(payload: dict) -> FieldValueCounts:
    """Flatten `payload` to a multiset of (index-free path, trimmed value) pairs.

    ``menu[0].nm`` and ``menu[3].nm`` both become ``menu[].nm``, so two identical
    rows contribute the same pair twice rather than two distinct pairs.

    Values are trimmed and stringified (matching the vendored evaluator's
    ``normalize_dict``, so a numeric leaf still matches its string counterpart on
    content — schema conformance is reported separately, §6.3). A value that is
    empty after trimming, and a value that is ``null``, count as absent.

    Note that TED-Acc does **not** apply the same empty rule to a whitespace-only
    leaf: the pinned ``normalize_dict`` tests falsiness *before* stripping, so
    ``"   "`` survives into its tree while it is dropped here. The divergence is
    characterized in ``tests/test_evaluation.py``; aligning them would mean
    patching the vendored evaluator, which ADR-022 declined to do.
    """
    field_values: FieldValueCounts = Counter()
    _collect_field_values(payload, path="", field_values=field_values)
    return field_values


def _collect_field_values(value: object, path: str, field_values: FieldValueCounts) -> None:
    """Accumulate `value`'s leaves into `field_values` under `path`."""
    if value is None:
        # §6.3: an absent key and a key present with `null` are identical.
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _collect_field_values(child, f"{path}.{key}" if path else str(key), field_values)
        return
    if isinstance(value, list):
        for item in value:
            _collect_field_values(item, f"{path}[]", field_values)
        return

    trimmed_value = str(value).strip()
    if trimmed_value:
        field_values[(path, trimmed_value)] += 1


def _loads_json_object(text: str) -> dict | None:
    """Parse `text` as JSON, returning it only if the top level is an object."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _unwrap_single_markdown_fence(text: str) -> str | None:
    """Return the text inside a fence wrapping the whole output, else ``None``.

    "Wrapped in exactly one fence" is taken literally: the output must open and
    close with a fence and contain no further fence between them, so a partial
    fence or prose surrounding a fenced block is not recoverable.
    """
    stripped_text = text.strip()
    if not stripped_text.startswith(MARKDOWN_FENCE) or not stripped_text.endswith(MARKDOWN_FENCE):
        return None
    if len(stripped_text) < 2 * len(MARKDOWN_FENCE):
        return None

    fenced_text = stripped_text[len(MARKDOWN_FENCE) : -len(MARKDOWN_FENCE)]
    if MARKDOWN_FENCE in fenced_text:
        return None

    # Whatever follows the opening fence on its own line is markdown's info
    # string (in practice `json`) and is never part of the payload. A fence with
    # no newline at all has no info string.
    info_string, newline, body = fenced_text.partition("\n")
    return body if newline else info_string


def _fraction(flags: Iterable[bool]) -> float:
    """Fraction of `flags` that are true; 0.0 when there is nothing to score."""
    observed_flags = list(flags)
    if not observed_flags:
        return 0.0
    return sum(observed_flags) / len(observed_flags)
