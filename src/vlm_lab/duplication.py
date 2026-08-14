"""The ADR-008 / ADR-019 / ADR-023 cross-split duplication audit.

The handling policy is **frozen before this code runs** (ADR-019 as amended by
ADR-021 and ADR-023, written up in ``docs/proposals/phase1_closure_prereg.md``
§8.1). This module *executes* that policy; it never chooses one. Nothing here
takes a threshold, a floor, or an action as an argument: they are module-level
constants precisely so that no caller can pick them once the counts are known.

The frozen algorithm, in one sentence: build **one** graph over all 1000
receipts, whose edges are equal exact-image hash, dHash Hamming distance
``<= 3``, or equal value-inclusive ground-truth hash; take connected components
**once**, before any exclusion; then exclude a `test` receipt whose component
contains any `train` or `validation` receipt, and a `validation` receipt whose
component contains any `train` receipt. `test`-only components are retained and
become the §6.4 bootstrap resampling units.

Two properties of that design are easy to lose in implementation, so they are
stated here:

**Exclusion propagates transitively.** Hamming adjacency is not transitive, so
``train — X — test`` chains exist in which the test receipt is nowhere near the
train receipt directly. ADR-023 pins the conservative reading: the whole
component goes. A pathological chain can therefore excise much of the test set,
which is not absorbed silently — it surfaces as the floors firing and the
comparison being declared ``NOT EVALUABLE``.

**The audit is published as aggregates only.** :func:`public_report` emits
counts and the verdict and nothing else. A test-side hash, row ID, or pair list
can be joined against an already-viewed train image and thereby reveal test
content, which is exactly what ADR-008 test blindness forbids. The per-receipt
detail on :class:`AuditResult` is the input to the §8.4 sealed exclusion
manifest, which stays sealed until Phase 5 completes; :class:`AuditResult`
itself is never a publishable object.

This module deliberately does not import :mod:`vlm_lab.data` or ``datasets``.
The caller loads the splits through
:func:`vlm_lab.mechanical_access.load_all_splits_for_mechanical_check` (which
logs the access) and converts the raw ground truths with
:func:`vlm_lab.data.convert_ground_truth` before handing content here.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from PIL import Image

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"
KNOWN_SPLIT_NAMES = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)

# Frozen edge signal (§8.1): dHash is 64-bit, computed from a 9x8 grayscale
# reduction, and two receipts are joined at Hamming distance <= 3.
DHASH_ROWS = 8
DHASH_COLUMNS = 8
MAX_DHASH_HAMMING_DISTANCE = 3
# Pinned so that the same image always yields the same dHash across runs and
# Pillow builds that change the default filter.
DHASH_RESAMPLING_FILTER = Image.Resampling.LANCZOS

# The ADR-023 NOT EVALUABLE floors. Named after the pre-registration's YAML keys
# (§6.4, §8.1) so a reported failure points straight at the frozen line.
MIN_RETAINED_RECEIPTS = 60
MIN_INDEPENDENT_CLUSTERS = 40
MIN_EFFECTIVE_SAMPLE_SIZE = 50.0
MIN_RETAINED_VALIDATION_RECEIPTS = 60

FLOOR_RETAINED_RECEIPTS = "min_retained_receipts"
FLOOR_INDEPENDENT_CLUSTERS = "min_independent_clusters"
FLOOR_EFFECTIVE_SAMPLE_SIZE = "min_effective_sample_size"
FLOOR_RETAINED_VALIDATION_RECEIPTS = "min_retained_validation_receipts"

VERDICT_EVALUABLE = "EVALUABLE"
VERDICT_NOT_EVALUABLE = "NOT EVALUABLE"

# The ADR-021 relation table's rows that can cause an exclusion. `test`↔`test`
# is absent by design: it never excludes anything.
TRAIN_TEST_RELATION = "train-test"
VALIDATION_TEST_RELATION = "validation-test"
TRAIN_VALIDATION_RELATION = "train-validation"
EXCLUSION_RELATIONS = (
    TRAIN_TEST_RELATION,
    VALIDATION_TEST_RELATION,
    TRAIN_VALIDATION_RELATION,
)

ReceiptId = str


@dataclass(frozen=True)
class ReceiptContent:
    """One receipt's audit inputs: its image and its converted ground truth.

    `converted_ground_truth` is the object returned by
    :func:`vlm_lab.data.convert_ground_truth`, not the raw JSON string.
    """

    image: Image.Image
    converted_ground_truth: dict


@dataclass(frozen=True)
class ReceiptSignals:
    """The frozen §8.1 signals for one receipt.

    `ground_truth_template_signature` is carried for reporting only: per ADR-023
    the type-only signature contributes **no** graph edges, because it would
    collapse independent receipts into a few enormous clusters.
    """

    receipt_id: ReceiptId
    split: str
    exact_image_hash: str
    perceptual_image_hash: int
    ground_truth_value_hash: str
    ground_truth_template_signature: str


@dataclass(frozen=True)
class ExcludedReceipt:
    """One excluded receipt and why it was excluded. Sealed detail, never published.

    `relations` can hold more than one entry: with transitive propagation a test
    receipt's component may contain both `train` and `validation` receipts.
    """

    receipt_id: ReceiptId
    split: str
    component_index: int
    relations: tuple[str, ...]


@dataclass(frozen=True)
class AuditResult:
    """The audit's full result. **Not publishable** — see :func:`public_report`.

    `retained_test_clusters` are the §6.4 bootstrap resampling units: the
    connected components made up entirely of `test` receipts.
    """

    retained_test_clusters: tuple[tuple[ReceiptId, ...], ...]
    retained_validation_receipts: int
    excluded_receipts: tuple[ExcludedReceipt, ...] = field(repr=False)

    @property
    def retained_test_receipts(self) -> int:
        return sum(len(cluster) for cluster in self.retained_test_clusters)

    @property
    def independent_test_clusters(self) -> int:
        return len(self.retained_test_clusters)

    @property
    def effective_sample_size(self) -> float:
        """Kish's ESS, ``(sum n_c)**2 / sum n_c**2``, over retained test cluster sizes.

        Zero when nothing is retained, which fails every floor anyway.
        """
        cluster_sizes = [len(cluster) for cluster in self.retained_test_clusters]
        sum_of_squares = sum(size**2 for size in cluster_sizes)
        if sum_of_squares == 0:
            return 0.0
        return sum(cluster_sizes) ** 2 / sum_of_squares

    @property
    def excluded_test_receipts(self) -> int:
        return self._excluded_count(TEST_SPLIT)

    @property
    def excluded_validation_receipts(self) -> int:
        return self._excluded_count(VALIDATION_SPLIT)

    @property
    def excluded_receipts_by_relation(self) -> dict[str, int]:
        """Excluded receipts per ADR-021 relation.

        A receipt excluded by a component holding both `train` and `validation`
        counts under both relations, so these do **not** sum to
        `excluded_test_receipts`; report that count separately.
        """
        return {
            relation: sum(
                1 for excluded in self.excluded_receipts if relation in excluded.relations
            )
            for relation in EXCLUSION_RELATIONS
        }

    @property
    def failed_floors(self) -> tuple[str, ...]:
        """The ADR-023 floors this result falls below, named by their YAML keys."""
        floor_checks = (
            (FLOOR_RETAINED_RECEIPTS, self.retained_test_receipts >= MIN_RETAINED_RECEIPTS),
            (
                FLOOR_INDEPENDENT_CLUSTERS,
                self.independent_test_clusters >= MIN_INDEPENDENT_CLUSTERS,
            ),
            (
                FLOOR_EFFECTIVE_SAMPLE_SIZE,
                self.effective_sample_size >= MIN_EFFECTIVE_SAMPLE_SIZE,
            ),
            (
                FLOOR_RETAINED_VALIDATION_RECEIPTS,
                self.retained_validation_receipts >= MIN_RETAINED_VALIDATION_RECEIPTS,
            ),
        )
        return tuple(name for name, holds in floor_checks if not holds)

    @property
    def verdict(self) -> str:
        """``NOT EVALUABLE`` if **any** ADR-023 floor fails, else ``EVALUABLE``.

        The three test floors and the validation floor have different
        consequences — the test floors block the confirmatory comparison, the
        validation floor halts Phase 4 checkpoint selection and escalates to the
        user — so read `failed_floors` to see which fired.
        """
        return VERDICT_NOT_EVALUABLE if self.failed_floors else VERDICT_EVALUABLE

    def _excluded_count(self, split: str) -> int:
        return sum(1 for excluded in self.excluded_receipts if excluded.split == split)


def exact_image_hash(image: Image.Image) -> str:
    """SHA-256 over the decoded RGB pixel bytes, the dimensions, and the mode.

    Decoded pixels rather than file bytes, because re-encoding the same receipt
    changes the file and not the picture. Dimensions and mode are hashed in
    alongside them (finding F4): the raw byte sequence of a 2x3 image equals
    that of a 3x2 image with the same pixels in the same order, and a grayscale
    image converted to RGB has the same bytes as an already-gray RGB image.
    """
    rgb_pixels = image.convert("RGB").tobytes()
    digest = hashlib.sha256()
    digest.update(f"{image.mode}|{image.width}x{image.height}|".encode("utf-8"))
    digest.update(rgb_pixels)
    return digest.hexdigest()


def perceptual_image_hash(image: Image.Image) -> int:
    """Return the 64-bit dHash of `image`.

    The image is reduced to a ``(DHASH_COLUMNS + 1) x DHASH_ROWS`` grayscale
    thumbnail and each pixel is compared with its right-hand neighbour, giving
    one bit per comparison. Row-major, most significant bit first, so the
    integer is a stable encoding of the same 64 comparisons every run.
    """
    grayscale = image.convert("L").resize(
        (DHASH_COLUMNS + 1, DHASH_ROWS), DHASH_RESAMPLING_FILTER
    )
    # One byte per pixel, because the thumbnail is mode "L".
    pixels = grayscale.tobytes()
    row_width = DHASH_COLUMNS + 1

    bits = 0
    for row in range(DHASH_ROWS):
        for column in range(DHASH_COLUMNS):
            left = pixels[row * row_width + column]
            right = pixels[row * row_width + column + 1]
            bits = (bits << 1) | int(left > right)
    return bits


def ground_truth_value_hash(converted_ground_truth: dict) -> str:
    """SHA-256 over a canonical serialization of the ground truth, values included.

    Detects genuinely duplicated annotations. Keys are sorted so the hash
    depends on content rather than insertion order; list order is preserved,
    because a receipt whose line items are reordered is a different receipt.
    """
    return _canonical_json_hash(converted_ground_truth)


def ground_truth_template_signature(converted_ground_truth: dict) -> str:
    """SHA-256 over the same canonical form with every leaf replaced by its type name.

    Detects shared *templates*. Reported separately as template evidence and
    **never** used as a graph edge (ADR-023): almost every CORD receipt shares a
    handful of structures, so this signal would merge independent receipts into
    a few enormous components.
    """
    return _canonical_json_hash(_replace_leaves_with_type_names(converted_ground_truth))


def build_duplication_graph(
    records: Sequence[ReceiptSignals],
) -> tuple[tuple[ReceiptId, ...], ...]:
    """Return the connected components of the one frozen cross-split graph.

    Vertices are every receipt of every split. An undirected edge joins two
    receipts when their exact image hashes are equal, **or** their dHash Hamming
    distance is at most :data:`MAX_DHASH_HAMMING_DISTANCE`, **or** their
    value-inclusive ground-truth hashes are equal. The template signature
    contributes no edge.

    Components are computed over the full graph, before any exclusion, so that
    the audit and the §6.4 bootstrap cannot disagree about cluster structure.
    Both the components and the receipts within them are returned in sorted
    order, so the result depends only on the input set.

    Raises:
        ValueError: if two records share a receipt ID.
    """
    receipt_ids = [record.receipt_id for record in records]
    duplicate_ids = sorted(
        receipt_id
        for receipt_id, occurrences in Counter(receipt_ids).items()
        if occurrences > 1
    )
    if duplicate_ids:
        raise ValueError(f"receipt IDs must be unique, but these repeat: {duplicate_ids}")

    components = _DisjointSets(receipt_ids)
    _union_equal_signal(components, records, lambda record: record.exact_image_hash)
    _union_equal_signal(components, records, lambda record: record.ground_truth_value_hash)
    _union_near_perceptual_hashes(components, records)
    return components.grouped_members()


def decide_exclusions(
    components: Sequence[Sequence[ReceiptId]],
    split_of: Mapping[ReceiptId, str],
) -> AuditResult:
    """Apply ADR-023's exclusion rule to already-computed components.

    A `test` receipt is excluded when its component contains any `train` or
    `validation` receipt; a `validation` receipt is excluded when its component
    contains any `train` receipt. Membership is what counts, so exclusion
    propagates transitively through the component — a `train — X — test` chain
    excludes the test receipt even though the two are not directly matched.
    `test`↔`test` links never exclude anything.

    Raises:
        ValueError: if `split_of` names an unknown split, or lacks an entry for
            a receipt in `components`.
    """
    _require_known_splits(split_of.values())

    retained_test_clusters: list[tuple[ReceiptId, ...]] = []
    retained_validation_receipts = 0
    excluded_receipts: list[ExcludedReceipt] = []

    for component_index, component in enumerate(components):
        splits_in_component = [
            (receipt_id, _split_of(split_of, receipt_id)) for receipt_id in component
        ]
        splits_present = {split for _, split in splits_in_component}
        test_exclusion_relations = _test_exclusion_relations(splits_present)

        for receipt_id, split in splits_in_component:
            if split == TEST_SPLIT and test_exclusion_relations:
                excluded_receipts.append(
                    ExcludedReceipt(
                        receipt_id, split, component_index, test_exclusion_relations
                    )
                )
            elif split == VALIDATION_SPLIT and TRAIN_SPLIT in splits_present:
                excluded_receipts.append(
                    ExcludedReceipt(
                        receipt_id, split, component_index, (TRAIN_VALIDATION_RELATION,)
                    )
                )
            elif split == VALIDATION_SPLIT:
                retained_validation_receipts += 1

        if splits_present == {TEST_SPLIT}:
            retained_test_clusters.append(tuple(component))

    return AuditResult(
        retained_test_clusters=tuple(retained_test_clusters),
        retained_validation_receipts=retained_validation_receipts,
        excluded_receipts=tuple(excluded_receipts),
    )


def audit_duplication(
    receipts_by_split: Mapping[str, Sequence[ReceiptContent]],
) -> AuditResult:
    """Run the whole audit: hash every receipt, build the graph, apply exclusions.

    `receipts_by_split` maps a split name to that split's receipts **in row
    order**; the row's position becomes its ID, so the IDs stay joinable with
    the dataset rows for the §8.4 sealed manifest.

    Raises:
        ValueError: if `receipts_by_split` names a split other than `train`,
            `validation` or `test`. Checked before any hashing, so a typo fails
            immediately rather than after a thousand images have been read.
    """
    _require_known_splits(receipts_by_split)
    records = [
        compute_receipt_signals(f"{split}:{row_index}", split, content)
        for split, receipts in receipts_by_split.items()
        for row_index, content in enumerate(receipts)
    ]
    components = build_duplication_graph(records)
    return decide_exclusions(components, {record.receipt_id: record.split for record in records})


def public_report(result: AuditResult) -> dict:
    """Render the audit as the aggregate counts and verdict that may be published.

    This is the **only** publishable view of the audit (§8.1 test blindness). It
    carries no hashes, no receipt IDs, no pair lists and no cluster membership,
    because a test-side hash joined against an already-viewed train image
    reveals test content. `retained_test_cluster_sizes` is a bare size multiset,
    which is what ADR-019 item 4 means by reporting the cluster structure and is
    what makes the reported ESS checkable.
    """
    return {
        "verdict": result.verdict,
        "failed_floors": list(result.failed_floors),
        "retained_test_receipts": result.retained_test_receipts,
        "excluded_test_receipts": result.excluded_test_receipts,
        "independent_test_clusters": result.independent_test_clusters,
        "retained_test_cluster_sizes": sorted(
            (len(cluster) for cluster in result.retained_test_clusters), reverse=True
        ),
        "effective_sample_size": result.effective_sample_size,
        "retained_validation_receipts": result.retained_validation_receipts,
        "excluded_validation_receipts": result.excluded_validation_receipts,
        "excluded_receipts_by_relation": result.excluded_receipts_by_relation,
        "floors": {
            FLOOR_RETAINED_RECEIPTS: MIN_RETAINED_RECEIPTS,
            FLOOR_INDEPENDENT_CLUSTERS: MIN_INDEPENDENT_CLUSTERS,
            FLOOR_EFFECTIVE_SAMPLE_SIZE: MIN_EFFECTIVE_SAMPLE_SIZE,
            FLOOR_RETAINED_VALIDATION_RECEIPTS: MIN_RETAINED_VALIDATION_RECEIPTS,
        },
    }


class _DisjointSets:
    """Union-find over receipt IDs, enough for connected components and no more."""

    def __init__(self, members: Sequence[ReceiptId]) -> None:
        self._parent = {member: member for member in members}

    def find(self, member: ReceiptId) -> ReceiptId:
        root = member
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[member] != root:
            self._parent[member], member = root, self._parent[member]
        return root

    def union(self, one: ReceiptId, other: ReceiptId) -> None:
        one_root, other_root = self.find(one), self.find(other)
        if one_root != other_root:
            self._parent[other_root] = one_root

    def grouped_members(self) -> tuple[tuple[ReceiptId, ...], ...]:
        """Return the components, each sorted, ordered by their first member."""
        members_by_root: dict[ReceiptId, list[ReceiptId]] = {}
        for member in self._parent:
            members_by_root.setdefault(self.find(member), []).append(member)
        return tuple(sorted(tuple(sorted(members)) for members in members_by_root.values()))


def compute_receipt_signals(
    receipt_id: ReceiptId, split: str, content: ReceiptContent
) -> ReceiptSignals:
    """Hash one receipt's image and ground truth into its small `ReceiptSignals`.

    `audit_duplication` calls this once per receipt and never touches
    `content.image` again afterward. Exposed publicly so a caller with 1000
    real, full-resolution receipt images can compute signals one receipt at a
    time and let each decoded image be garbage-collected immediately, instead
    of decoding and holding all of them at once just to satisfy
    `audit_duplication`'s `Sequence[ReceiptContent]` contract -- materializing
    1000 real CORD v2 images simultaneously is enough to exhaust a Colab
    session's system RAM (see `docs/STATE.md`'s notes on this). Callers that
    can afford to hold everything in memory (e.g. small test fixtures) should
    keep using `audit_duplication` directly; this function plus
    `build_duplication_graph` and `decide_exclusions` are the same algorithm,
    decomposed for streaming.
    """
    return ReceiptSignals(
        receipt_id=receipt_id,
        split=split,
        exact_image_hash=exact_image_hash(content.image),
        perceptual_image_hash=perceptual_image_hash(content.image),
        ground_truth_value_hash=ground_truth_value_hash(content.converted_ground_truth),
        ground_truth_template_signature=ground_truth_template_signature(
            content.converted_ground_truth
        ),
    )


def _union_equal_signal(
    components: _DisjointSets,
    records: Sequence[ReceiptSignals],
    signal_of: Callable[[ReceiptSignals], str],
) -> None:
    """Join every pair of receipts sharing one exact-equality signal's value."""
    first_receipt_by_signal: dict[str, ReceiptId] = {}
    for record in records:
        signal = signal_of(record)
        first_receipt = first_receipt_by_signal.setdefault(signal, record.receipt_id)
        components.union(first_receipt, record.receipt_id)


def _union_near_perceptual_hashes(
    components: _DisjointSets, records: Sequence[ReceiptSignals]
) -> None:
    """Join every pair of receipts within the frozen dHash Hamming distance.

    All pairs are compared. At CORD v2's 1000 receipts that is half a million
    64-bit XORs, which is not worth an index that could change which pairs are
    found.
    """
    for position, record in enumerate(records):
        for other in records[position + 1 :]:
            distance = (record.perceptual_image_hash ^ other.perceptual_image_hash).bit_count()
            if distance <= MAX_DHASH_HAMMING_DISTANCE:
                components.union(record.receipt_id, other.receipt_id)


def _test_exclusion_relations(splits_present: set[str]) -> tuple[str, ...]:
    """Name the relations that exclude the `test` receipts of a component."""
    relations = []
    if TRAIN_SPLIT in splits_present:
        relations.append(TRAIN_TEST_RELATION)
    if VALIDATION_SPLIT in splits_present:
        relations.append(VALIDATION_TEST_RELATION)
    return tuple(relations)


def _require_known_splits(split_names: Iterable[str]) -> None:
    unknown_splits = sorted(set(split_names) - set(KNOWN_SPLIT_NAMES))
    if unknown_splits:
        raise ValueError(
            f"unknown split name(s) {unknown_splits}; expected one of {list(KNOWN_SPLIT_NAMES)}"
        )


def _split_of(split_of: Mapping[ReceiptId, str], receipt_id: ReceiptId) -> str:
    if receipt_id not in split_of:
        raise ValueError(f"no split recorded for receipt {receipt_id!r}")
    return split_of[receipt_id]


def _canonical_json_hash(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _replace_leaves_with_type_names(value: object) -> object:
    if isinstance(value, dict):
        return {key: _replace_leaves_with_type_names(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_leaves_with_type_names(item) for item in value]
    return type(value).__name__
