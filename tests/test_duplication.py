"""Unit tests for src/vlm_lab/duplication.py, the ADR-008/019/023 duplication audit.

The audit executes a frozen policy, so these tests pin the policy's edges rather
than the implementation's convenience: transitive propagation of cross-split
exclusion, `test`↔`test` retention, the three test floors plus the validation
floor, and the fact that the publishable report leaks neither hashes nor row IDs.

Graph and exclusion behaviour is exercised through hand-written
:class:`ReceiptSignals`, because a real image cannot be made to sit at an exact
Hamming distance from another one on demand, and the distances are the whole
point of the transitive case. The hash signals themselves, and the end-to-end
runner, use small synthetic PIL images. No dataset is downloaded.
"""

from __future__ import annotations

import json
import random
import re

import pytest
from PIL import Image

from vlm_lab.duplication import (
    FLOOR_EFFECTIVE_SAMPLE_SIZE,
    MAX_DHASH_HAMMING_DISTANCE,
    TRAIN_TEST_RELATION,
    TRAIN_VALIDATION_RELATION,
    VALIDATION_TEST_RELATION,
    VERDICT_NOT_EVALUABLE,
    AuditResult,
    ReceiptContent,
    ReceiptSignals,
    audit_duplication,
    build_duplication_graph,
    decide_exclusions,
    exact_image_hash,
    ground_truth_template_signature,
    ground_truth_value_hash,
    perceptual_image_hash,
    public_report,
)

# dHash values chosen so that every pairwise Hamming distance below is explicit.
# The first three form the transitive chain: A—A_NEAR and A_NEAR—B are both
# within the frozen distance of 3, while A—B is 6 apart and therefore unmatched.
DHASH_A = 0b0
DHASH_A_NEAR = 0b111
DHASH_B = 0b111111
DHASH_ISOLATED = 0xFFFFFFFF
DHASH_ISOLATED_OTHER = 0xFFFFFFFF00000000


def signals(
    receipt_id: str,
    split: str,
    perceptual_hash: int,
    *,
    image_hash: str | None = None,
    value_hash: str | None = None,
    template_signature: str = "shared-template",
) -> ReceiptSignals:
    """Build one receipt's signals, unique by default so it links to nothing.

    `perceptual_hash` is always explicit: it is the signal whose distances the
    graph tests are about. The exact-image and ground-truth hashes default to
    per-receipt values, so a test that wants such an edge has to ask for it.
    """
    return ReceiptSignals(
        receipt_id=receipt_id,
        split=split,
        exact_image_hash=image_hash or f"image-hash-of-{receipt_id}",
        perceptual_image_hash=perceptual_hash,
        ground_truth_value_hash=value_hash or f"value-hash-of-{receipt_id}",
        ground_truth_template_signature=template_signature,
    )


def split_of(*records: ReceiptSignals) -> dict[str, str]:
    return {record.receipt_id: record.split for record in records}


def noise_image(seed: int, size_px: tuple[int, int] = (16, 16)) -> Image.Image:
    """A deterministic pseudo-random RGB image.

    Random rather than flat, because a uniform image has an all-zero dHash and
    every uniform image would then match every other one.
    """
    width_px, height_px = size_px
    pixel_bytes = random.Random(seed).randbytes(width_px * height_px * 3)
    return Image.frombytes("RGB", size_px, pixel_bytes)


def test_exact_image_hash_is_equal_for_separately_built_identical_images():
    assert exact_image_hash(noise_image(1)) == exact_image_hash(noise_image(1))


def test_exact_image_hash_differs_for_identical_bytes_at_different_dimensions():
    pixel_bytes = bytes(range(18))  # six RGB pixels

    assert exact_image_hash(Image.frombytes("RGB", (2, 3), pixel_bytes)) != exact_image_hash(
        Image.frombytes("RGB", (3, 2), pixel_bytes)
    )


def test_exact_image_hash_differs_for_identical_rgb_pixels_in_different_modes():
    gray_levels = bytes([10, 20, 30, 40])
    grayscale_image = Image.frombytes("L", (2, 2), gray_levels)
    already_rgb_image = Image.frombytes(
        "RGB", (2, 2), bytes(level for level in gray_levels for _ in range(3))
    )

    assert grayscale_image.convert("RGB").tobytes() == already_rgb_image.tobytes()
    assert exact_image_hash(grayscale_image) != exact_image_hash(already_rgb_image)


def test_perceptual_image_hash_is_a_stable_64_bit_value():
    hash_value = perceptual_image_hash(noise_image(2))

    assert 0 <= hash_value < 2**64
    assert hash_value == perceptual_image_hash(noise_image(2))


def test_ground_truth_value_hash_ignores_key_order_but_not_values():
    assert ground_truth_value_hash({"total": "12,000", "menu": []}) == ground_truth_value_hash(
        {"menu": [], "total": "12,000"}
    )
    assert ground_truth_value_hash({"total": "12,000"}) != ground_truth_value_hash(
        {"total": "13,000"}
    )


def test_ground_truth_template_signature_ignores_values_but_not_structure():
    assert ground_truth_template_signature(
        {"menu": [{"nm": "Latte"}], "total": "12,000"}
    ) == ground_truth_template_signature({"menu": [{"nm": "Bagel"}], "total": "13,000"})
    assert ground_truth_template_signature({"total": "12,000"}) != (
        ground_truth_template_signature({"total": "12,000", "cashprice": "12,000"})
    )


def test_shared_template_signature_alone_does_not_create_an_edge():
    """ADR-023: the type-only signature is template evidence, never a graph edge."""
    train_receipt = signals("train:0", "train", DHASH_ISOLATED)
    test_receipt = signals("test:0", "test", DHASH_ISOLATED_OTHER)

    components = build_duplication_graph([train_receipt, test_receipt])

    assert components == (("test:0",), ("train:0",))


def test_equal_exact_image_hash_creates_an_edge():
    train_receipt = signals("train:0", "train", DHASH_ISOLATED, image_hash="same-photo")
    test_receipt = signals("test:0", "test", DHASH_ISOLATED_OTHER, image_hash="same-photo")

    assert build_duplication_graph([train_receipt, test_receipt]) == (("test:0", "train:0"),)


def test_equal_ground_truth_value_hash_creates_an_edge():
    train_receipt = signals("train:0", "train", DHASH_ISOLATED, value_hash="same-annotation")
    test_receipt = signals("test:0", "test", DHASH_ISOLATED_OTHER, value_hash="same-annotation")

    assert build_duplication_graph([train_receipt, test_receipt]) == (("test:0", "train:0"),)


def test_test_receipt_is_excluded_through_an_intermediate_it_alone_does_not_match():
    """ADR-023's propagation rule, in the case where direct-pair matching differs.

    `train:0 — test:0 — test:1` is a chain: each link is within the frozen dHash
    distance of 3, but `train:0` and `test:1` are 6 apart, so `test:1` has no
    direct match in train or validation at all. Direct-pair matching would
    retain it; ADR-023 excludes the whole component, because a
    `train — X — test` chain still indicates shared template lineage.
    """
    train_receipt = signals("train:0", "train", DHASH_A)
    intermediate_test_receipt = signals("test:0", "test", DHASH_A_NEAR)
    chained_test_receipt = signals("test:1", "test", DHASH_B)

    assert (DHASH_A ^ DHASH_A_NEAR).bit_count() <= MAX_DHASH_HAMMING_DISTANCE
    assert (DHASH_A_NEAR ^ DHASH_B).bit_count() <= MAX_DHASH_HAMMING_DISTANCE
    assert (DHASH_A ^ DHASH_B).bit_count() > MAX_DHASH_HAMMING_DISTANCE

    records = [train_receipt, intermediate_test_receipt, chained_test_receipt]
    result = decide_exclusions(build_duplication_graph(records), split_of(*records))

    assert result.retained_test_receipts == 0
    assert result.excluded_test_receipts == 2
    assert result.excluded_receipts_by_relation[TRAIN_TEST_RELATION] == 2


def test_test_to_test_duplicates_are_retained_in_one_cluster():
    records = [
        signals("test:0", "test", DHASH_A),
        signals("test:1", "test", DHASH_A_NEAR),
        signals("test:2", "test", DHASH_ISOLATED),
    ]

    result = decide_exclusions(build_duplication_graph(records), split_of(*records))

    assert result.excluded_test_receipts == 0
    assert result.retained_test_receipts == 3
    assert result.retained_test_clusters == (("test:0", "test:1"), ("test:2",))


def test_validation_receipt_is_excluded_when_its_component_contains_train():
    records = [
        signals("train:0", "train", DHASH_A),
        signals("validation:0", "validation", DHASH_A_NEAR),
        signals("validation:1", "validation", DHASH_ISOLATED),
    ]

    result = decide_exclusions(build_duplication_graph(records), split_of(*records))

    assert result.excluded_validation_receipts == 1
    assert result.retained_validation_receipts == 1
    assert result.excluded_receipts_by_relation[TRAIN_VALIDATION_RELATION] == 1


def test_test_receipt_is_excluded_when_its_component_contains_validation():
    """Validation drives prompt and checkpoint selection, so it contaminates test too.

    The validation receipt in that same component is itself retained: only a
    `train` link excludes a validation receipt.
    """
    records = [
        signals("validation:0", "validation", DHASH_A),
        signals("test:0", "test", DHASH_A_NEAR),
    ]

    result = decide_exclusions(build_duplication_graph(records), split_of(*records))

    assert result.excluded_test_receipts == 1
    assert result.excluded_receipts_by_relation[VALIDATION_TEST_RELATION] == 1
    assert result.excluded_receipts_by_relation[TRAIN_TEST_RELATION] == 0
    assert result.retained_validation_receipts == 1


def test_thirty_nine_singletons_and_one_large_cluster_fail_only_the_ess_floor():
    """The §6.4 / ADR-023 worked example: count floors cleared, ESS 7.5, NOT EVALUABLE."""
    singleton_clusters = [(f"test:{index}",) for index in range(39)]
    large_cluster = tuple(f"test:{index}" for index in range(39, 60))
    validation_receipts = [(f"validation:{index}",) for index in range(60)]
    components = [*singleton_clusters, large_cluster, *validation_receipts]
    splits = {
        receipt_id: receipt_id.split(":")[0]
        for component in components
        for receipt_id in component
    }

    result = decide_exclusions(components, splits)

    assert result.retained_test_receipts == 60
    assert result.independent_test_clusters == 40
    assert result.effective_sample_size == 60**2 / (39 * 1**2 + 21**2)
    assert result.effective_sample_size == 7.5
    assert result.retained_validation_receipts == 60
    assert result.failed_floors == (FLOOR_EFFECTIVE_SAMPLE_SIZE,)
    assert result.verdict == VERDICT_NOT_EVALUABLE


def test_public_report_publishes_no_hashes_receipt_ids_or_cluster_membership():
    result = AuditResult(
        retained_test_clusters=(("test:7", "test:8"), ("test:9",)),
        retained_validation_receipts=61,
        excluded_receipts=decide_exclusions(
            [("train:3", "test:3")], {"train:3": "train", "test:3": "test"}
        ).excluded_receipts,
    )

    report = public_report(result)
    serialized_report = json.dumps(report)

    for leaked_receipt_id in ("test:7", "test:8", "test:9", "test:3", "train:3"):
        assert leaked_receipt_id not in serialized_report
    assert not re.search(r"[0-9a-f]{16,}", serialized_report)
    assert _string_values_in(report) == {
        VERDICT_NOT_EVALUABLE,
        "min_retained_receipts",
        "min_independent_clusters",
        "min_effective_sample_size",
        "min_retained_validation_receipts",
        TRAIN_TEST_RELATION,
        VALIDATION_TEST_RELATION,
        TRAIN_VALIDATION_RELATION,
        "verdict",
        "failed_floors",
        "retained_test_receipts",
        "excluded_test_receipts",
        "independent_test_clusters",
        "retained_test_cluster_sizes",
        "effective_sample_size",
        "retained_validation_receipts",
        "excluded_validation_receipts",
        "excluded_receipts_by_relation",
        "floors",
    }


def test_audit_duplication_hashes_real_images_and_excludes_a_duplicated_test_row():
    shared_image = noise_image(seed=100)
    receipts_by_split = {
        "train": [ReceiptContent(shared_image, {"total": "1,000"})],
        "validation": [ReceiptContent(noise_image(seed=200), {"total": "2,000"})],
        "test": [
            ReceiptContent(shared_image, {"total": "3,000"}),
            ReceiptContent(noise_image(seed=300), {"total": "4,000"}),
        ],
    }

    result = audit_duplication(receipts_by_split)

    assert result.excluded_test_receipts == 1
    assert result.excluded_receipts_by_relation[TRAIN_TEST_RELATION] == 1
    assert result.retained_test_clusters == (("test:1",),)
    assert result.retained_validation_receipts == 1
    # Far below every floor: a two-row test set cannot support a held-out claim.
    assert result.verdict == VERDICT_NOT_EVALUABLE


def test_an_unknown_split_name_is_rejected_rather_than_silently_ignored():
    with pytest.raises(ValueError, match="unknown split name"):
        decide_exclusions([("dev:0",)], {"dev:0": "dev"})


def test_repeated_receipt_ids_are_rejected():
    duplicated_id = [
        signals("test:0", "test", DHASH_A),
        signals("test:0", "test", DHASH_ISOLATED),
    ]

    with pytest.raises(ValueError, match="receipt IDs must be unique"):
        build_duplication_graph(duplicated_id)


def _string_values_in(value: object) -> set[str]:
    """Every string appearing anywhere in `value`, keys included."""
    if isinstance(value, dict):
        return set(value) | {
            found for item in value.values() for found in _string_values_in(item)
        }
    if isinstance(value, list):
        return {found for item in value for found in _string_values_in(item)}
    return {value} if isinstance(value, str) else set()
