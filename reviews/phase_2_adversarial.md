# Phase 2 Entry — ADR-005 Adversarial Review

- **Review date:** 2026-08-12
- **Reviewed phase:** Phase 1 closure → Phase 2 entry gate (ADR-005)
- **Reviewed state:** commit `f179dbd` on branch `worktree-phase2-gate`; review target
  `docs/proposals/phase1_closure_prereg.md` (draft proposal), read against `AGENTS.md`,
  `docs/EXPERIMENT_SPEC.md`, `docs/EVALUATION_PROTOCOL.md`, `docs/IMPLEMENTATION_PLAN.md`,
  `docs/DECISIONS.md`, `docs/STATE.md`, `src/vlm_lab/data.py`, `notebooks/`, `pyproject.toml`
- **Review model:** Codex (GPT-5.6 Sol role), read-only, independent context
- **Review mode:** Adversarial (design/methodology), per `AGENTS.md` §14.2
- **Invocation:** `codex:rescue` skill — this repository's stand-in for `/codex:review`
  (project memory `delegation_tooling_mapping`). Working tree was clean and unmodified after the
  review; the reviewer made no repository changes.

## Overall review outcome

**BLOCK — the ADR-005 gate is NOT satisfied. Phase 2 must not start.**

25 findings: 17 `required`, 7 `recommended`, 1 `optional`.

The reviewer's summary of the failure mode: the design's skeleton is sound, but four classes of hole
remain through which a result could be made to look better (or worse) than it is — duplicate data,
metric–task misalignment, truncation, and unfrozen training conditions.

## Claude Code disposition summary

Every one of the 25 findings is **ACCEPT**. This is not deference: each was checked against the
proposal text and, where arithmetic or repository facts were involved, re-verified independently.
Two of them identify outright errors in Claude Code's own reasoning (F3 and F23 below), and several
others identify claims that were stated with more confidence than the evidence supported. No finding
was found to be technically incorrect, so no `REJECT` is recorded.

Four decisions are escalated as `USER DECISION REQUIRED` (`AGENTS.md` §17): two were already open
before this review, two are newly created by it.

| Class | Count |
|---|---|
| ACCEPT | 25 |
| REJECT | 0 |
| DEFER | 0 |
| USER DECISION REQUIRED (escalated, arising from the above) | 4 |

---

## Findings and dispositions

### Fairness

**F1 — "Byte-identical prompt" does not freeze the model input.** `required`
The proposal pinned the prompt *string* but left `apply_chat_template` arguments, image/text
content ordering, `add_generation_prompt`, target JSON serialization, output slicing, padding and
truncation sides, EOS/PAD handling, `num_beams`, `use_cache`, cache implementation, and stopping
behaviour unspecified. Identical strings can still yield different token prefixes.
→ **ACCEPT.** Correct and material: this is precisely the ADR-006 asymmetry the project exists to
prevent, and every one of these knobs is currently selectable *after* seeing validation output.
Pre-registration must cover the fully rendered message construction, the complete `GenerationConfig`,
processor kwargs, and the decoding procedure, with an assertion that both conditions receive
identical token IDs, masks and image grids.

**F2 — `max_seq_len: 2048` is inconsistent with the proposed caps.** `required`
1,024 image tokens + 768 `max_new_tokens` leaves 256 tokens for the system prompt, user prompt,
image boundary tokens and chat-template tokens. Training could silently drop target suffixes and
EOS; evaluation could cut a correct JSON object before its closing brace and score it zero.
→ **ACCEPT.** A genuine defect introduced by fixing §4's `max_seq_len` and §5.3's `max_new_tokens`
independently of each other. The budget must be derived jointly from the rendered train+validation
corpus, with zero training-target truncations and zero silent input truncations as hard assertions.
Hitting `max_new_tokens` at test time is model behaviour, not an execution defect — it must not
become grounds for a rerun with a larger cap.

### Contamination

**F3 — "A duplicate inflates both conditions, not one" is FALSE.** `required`
The Fine-tuned condition has *trained* on a train↔test duplicate; the Base condition has not.
Pairing the two predictions on the same test row does not remove that training exposure. The
proposal's report-only default also violates ADR-008, which requires the handling policy to be
pre-registered *before* the audit is run.
→ **ACCEPT — this is the most serious finding, and it is an error in Claude Code's own reasoning.**
The proposal's justification for report-only was wrong on the merits: memorized duplicates
selectively raise the Fine-tuned score and manufacture exactly the apparent improvement the held-out
split exists to rule out. The policy must be frozen before the audit runs, with deterministic actions
pre-specified for both exact and near duplicates. **The choice of policy is escalated as
`USER DECISION REQUIRED` (D-3)** — it is a methodology decision, and any departure from ADR-007's
official split requires its own ADR.

**F4 — Test-side hashes are themselves a leak, and validation↔test duplication is unaudited.**
`required`
A published test-side hash can be joined against an already-viewed train/validation image, revealing
which receipt is in test without ever rendering it. Validation drives prompt and checkpoint
selection, so a validation↔test duplicate is also contamination. The proposed structure hash
replaces every leaf value with its type, so it detects shared *layouts*, not duplicate annotations.
→ **ACCEPT.** Before Phase 5 the audit must expose only aggregate counts and a frozen automated
verdict — no hashes, pair IDs or candidate renderings. It must cover train↔test, validation↔test and
train↔validation, add a value-inclusive canonical ground-truth hash alongside the type-only
signature, and include image dimensions and mode in the exact pixel hash.

**F5 — Test-blindness is not structurally enforced.** `required`
`load_cord_v2()` returns a `DatasetDict` containing all three splits, so the proposed Phase 2
acceptance criterion ("the test split is not read") depends on caller discipline.
→ **ACCEPT.** Requires a split-scoped loader or explicit allowed-splits API that fails closed on
`"test"`, plus a test proving Phase 2 issues no test-split request. This modifies Phase 1 code
(`src/vlm_lab/data.py`) and is therefore Phase 1 closure work, not Phase 2 work.

**F6 — "Truncated run" is an overly broad test-rerun exception.** `required`
It could equally mean an infrastructure interruption or outputs simply reaching `max_new_tokens`.
A Base-complete / Fine-tuned-failed run would also expose Base test results before the second
condition is debugged.
→ **ACCEPT.** Infrastructure interruption must be defined separately from generation-length
termination; only an exact-config replay is permitted for infrastructure or artifact failure;
outputs stay sealed until both conditions and the artifact-integrity checks finish; every attempted
test access is recorded.

### Metrics

**F7 — A key inventory is not a normative schema.** `required`
Allowed types, unknown keys, scalar-versus-list shape, nested `sub` rules and strict schema validity
remain undefined. Donut's evaluator stringifies scalar leaves and drops some empty values, so a
schema-invalid prediction can still receive content credit.
→ **ACCEPT.** A recursive schema specification must be frozen from train+validation only, with raw
JSON validity and strict schema validity reported separately.

**F8 — TED-Acc's alignment with the task is asserted, not established.** `required`
It was selected because it is the existing "leading candidate" and yields a per-receipt score.
But Donut's TED uses character edit cost, so `58,000` versus `59,000` — a materially wrong
amount — can cost a single character edit. TED-Acc can therefore improve without the extracted
values improving.
→ **ACCEPT.** A cheap, test-free probe must run *before any model output exists*: score both
candidate metrics on synthetic error cases (wrong amount digit, missing field, extra field, item
reorder, invalid JSON, near-correct OCR). **The resulting choice of primary metric is escalated as
`USER DECISION REQUIRED` (D-4)** — changing the primary metric is a material evaluation-methodology
decision under `AGENTS.md` §17.

**F9 — NFKC and whitespace collapse contradict the prompt's "copy values verbatim".** `required`
The proposal rejects digit-grouping normalization because formatting is real signal, then discards
other formatting distinctions. The instruction and the metric are measuring different tasks.
→ **ACCEPT.** One estimand must be chosen — strict transcription (preserve Unicode and internal
whitespace; report a normalized diagnostic separately) or semantic extraction (redefine the prompt
and enumerate every accepted equivalence). Folded into D-4, since it is the same choice.

**F10 — Fence-stripped output is counted as raw-valid JSON.** `required`
The prompt forbids fences, so fenced output is not a JSON document. If the Base model fences more
often than the Fine-tuned model, silent recovery materially changes the measured adapter effect.
→ **ACCEPT.** Strict raw JSON validity and recoverable-payload parse success must be reported
separately, with the representation used for content metrics pre-registered.

**F11 — "Vendoring at a pin means there are no differences" is false.** `required`
The proposal adds pre-normalization, parsing, invalid-output and missing/null rules *around* Donut,
making it a custom pipeline. It also names only `zss`, while official `donut/util.py` imports
`nltk.edit_distance` directly.
→ **ACCEPT**, including the `nltk` catch, which is a concrete factual correction. Pin the Donut
commit, `zss` and `nltk`; document every wrapper difference; vendor only the required code with its
license; freeze regression fixtures with known scores.

**F12 — "Micro-F1 cannot be a bootstrap statistic" is incorrect.** `recommended`
It is incompatible with the proposed *mean-of-per-receipt-differences* construction, but a paired
receipt-level cluster bootstrap can resample receipts and recompute the corpus statistic per
replicate.
→ **ACCEPT.** Wording correction; optionally provide a cluster-bootstrap CI if micro-F1 is retained
as a secondary metric. Note this weakens one of the stated reasons for preferring TED-Acc, which
feeds D-4.

**F13 — I.I.D. receipt resampling is invalid if the test set contains template clusters.**
`required` (`[HYPOTHESIS]`, correctly labelled)
Correlated receipts resampled as independent evidence narrow the CI and raise the false
"improvement achieved" rate. The proposal's own structure audit acknowledges template similarity but
never connects it to the resampling assumption.
→ **ACCEPT.** A group-aware bootstrap or exclusion rule must be pre-registered against the frozen
duplication clusters; if the audit shows no meaningful clusters, the ordinary paired bootstrap is
retained and that evidence is recorded. Depends on D-3.

### Pre-registration completeness

**F14 — "Everything except the two knobs below is fixed" is false.** `required`
Unspecified: optimizer and optimizer-state precision, scheduler, warmup, weight decay, gradient
clipping, LoRA init / RSLoRA / DoRA flags, packing, target serialization, evaluation batch size,
save schedule. `peft` and `bitsandbytes` are absent from `pyproject.toml` entirely.
→ **ACCEPT.** Verified directly: `pyproject.toml` pins `datasets`, `transformers`, `torch`,
`pillow`, `torchvision` and does not contain `peft` or `bitsandbytes`. The complete training
configuration must be frozen before Phase 2, with those and any training runtime pinned, plus a
resolved-environment or lock artifact — exact `==` pins alone do not freeze transitive resolution
across Colab sessions.

**F15 — The role of the validation split is ambiguous, and `IMPLEMENTATION_PLAN.md` contradicts
itself.** `required`
Phase 4's description says `qwen_cord_mini` references only validation, while the configs table says
"train subset + validation". Training on validation and then selecting checkpoints by validation
TED-Acc would destroy the selection holdout.
→ **ACCEPT.** This is a pre-existing defect in an authoritative document, not only in the proposal,
and must be corrected there. Exact split roles must be stated: every trial fits on train only,
evaluates on untouched validation, and never updates weights from validation.

**F16 — The draft is not yet a complete pre-registration, and contains unsupported predictions.**
`required`
Still open: X, Colab tier, Donut SHA, metric dependency versions, schema inventory, token
distributions, final `max_new_tokens`, duplication outcome and policy, VRAM verdict. The threshold
discussion also asserts "almost certain to pass" and "typically moves schema-adherence a long way"
without evidence.
→ **ACCEPT**, including the criticism of the unsupported predictions — in a pre-registration
document, forecasting the result is exactly the wrong move, and X must be chosen from what a TED-Acc
delta *means*, not from how likely it is to pass. Those sentences are removed.

**F17 — Aggregate `metrics.json` / `report.md` cannot support an audit of paired statistics.**
`required`
A wrong join, a dropped parse failure or a duplicated sample still yields plausible aggregates.
→ **ACCEPT.** Pre-register an immutable per-sample artifact (stable sample ID, condition, raw
output, termination reason, parse status, parsed object, reference linkage/hash, every per-sample
metric, exception state, config/artifact hashes) and assert one-to-one Base/Fine-tuned/reference
alignment before bootstrapping.

### Factual errors

**F18 — 1,093 and 1,599 are raw area divisions, not processor token counts.** `recommended`
Qwen's image processor rounds both dimensions to multiples of `patch_size × merge_size = 32`, giving
roughly 1,080 and 1,610 merged tokens before caps.
→ **ACCEPT.** Counts must come from the pinned processor's actual `image_grid_thw`, not arithmetic.
Related correction found independently by Claude Code while the review ran: the pinned processor
takes `size: {"longest_edge", "shortest_edge"}` in **total pixels** (its real Colab output shows
`16777216` / `65536`), so the proposal's `image_max_pixels` must be mapped onto those keys
explicitly rather than invented as new parameter names.

**F19 — "No shared module basename" is too broad; "no separate `lm_head`" is imprecise.** `optional`
Both towers contain modules named `norm`. Qwen3-VL does have an `lm_head` module; it is its
*weight* that is tied, so no separate `lm_head.weight` tensor appears in the checkpoint.
→ **ACCEPT.** Does not invalidate the proposed regex, but the ADR-012 evidence must be stated
narrowly: none of the seven selected *projection* basenames occurs in the vision tower, and the
output-head parameter is tied rather than the module being absent.

**F20 — `attn_implementation="sdpa"` does not pin the underlying kernel; the load dtype of
non-quantized modules is unpinned.** `recommended`
SDPA dispatches among available backends, so a tier change alters the numerical path while the YAML
still says `sdpa`. Leaving non-quantized modules on the config's BF16 undermines the intended T4 FP16
path.
→ **ACCEPT.** Pin the model load dtype explicitly, record all realized parameter/module dtypes, and
record hardware plus the selected SDPA backend, keeping the realized path identical for both
conditions.

### VRAM

**F21 — The 13.0 GiB criterion is not shown to cover evaluation.** `required`
`max_memory_reserved()` during a training step says nothing about generation prefill/KV-cache peaks
or non-PyTorch GPU allocations; the 1.56 GiB residual is an unsupported margin; and the optimizer
itself is unspecified (FP32 Adam moments for 33,030,144 parameters are ~252 MiB before gradients,
master weights, activations and allocator effects).
→ **ACCEPT.** Separate gates are needed for model load/quantization, training across a full
accumulation window with the exact optimizer, checkpoint save/resume, and Base/Fine-tuned generation
at maximum input plus maximum output length — measured against the selected tier's actual free
memory across the full 900-row train+validation token distribution.

### Scope

**F22 — "No condition-dependent branch anywhere" contradicts the one authorized difference.**
`recommended`
Adapter enabled versus disabled *is* a condition-dependent difference; an impossible acceptance
criterion invites superficial compliance.
→ **ACCEPT.** Allowlist exactly one difference — adapter state — and assert a structured diff of all
resolved inputs and configuration, preferring a single quantized base instance with a verified PEFT
adapter enable/disable path.

### Dismissed alternatives

**F23 — The few-shot dismissal invoked optimizer-state memory, but inference has no optimizer.**
`recommended`
→ **ACCEPT — a second error in Claude Code's own reasoning.** The stated rationale conflated
training-time and inference-time memory. Few-shot may still be infeasible on account of image-context
and KV-cache memory, but that is now a `[HYPOTHESIS]` to be measured in the evaluation VRAM gate, not
an established fact. The claim must be relabelled as "adapter effect under the frozen zero-shot
prompt" rather than presented as a general Base-versus-Fine-tuned comparison.

**F24 — "BCa's correction is second-order at n=100" is unsupported.** `recommended`
Sample size alone does not establish negligible bias/skew correction for a bounded, potentially
zero-inflated TED-difference distribution.
→ **ACCEPT.** Percentile is retained *for transparency and simplicity*, labelled honestly as that
trade-off, or compared against BCa on synthetic / train+validation surrogate scores before freezing.

**F25 — "The task is not learn to see receipts" is a hypothesis, not a finding.** `recommended`
No full-model GPU inference has been run and no performance output observed, so the vision tower's
adequacy is unverified. Language-only LoRA narrows the experiment from general QLoRA adaptation to
decoder-side adaptation.
→ **ACCEPT.** Relabel as a hypothesis and a budget-driven exclusion, and scope the Phase 6
conclusion accordingly. Vision/projector LoRA remains out of the current experiment.

---

## Parts the reviewer probed and found defensible

Recorded so that later sessions do not re-litigate them:

- The proposal correctly remains non-authoritative and forbids Phase 2 execution before promotion.
- Runtime quantization of one pinned Qwen artifact is a defensible shared foundation for both
  conditions (once `bitsandbytes` and the realized configs are pinned — F14, F20).
- The LoRA arithmetic is correct: `57,344 × 36 × 16 = 33,030,144`.
- PEFT does accept a regex string for `target_modules`, and the proposed regex isolates the language
  projections.
- Qwen3-VL's embedding / output-head weight is genuinely tied.
- A merged visual token corresponds conceptually to a 32×32 pixel block.
- T4 lacks BF16 Tensor Core support, and the standard FlashAttention-2 CUDA implementation does not
  support Turing; FP16 plus an SDPA fallback is defensible.
- Assistant-only label masking is appropriate.
- Greedy decoding and a shared inference/evaluation path are defensible.
- Receipt-level TED-Acc differences make the proposed paired mean-difference bootstrap well defined.
- Not normalizing digit grouping is defensible under the stated verbatim-value objective.
- Pipeline completion and model-performance improvement remain correctly separated.

## Reviewer's stated limitations

- No GPU or model code was executed; module injection, token distributions, duplicates and memory
  peaks remain open empirical gates.
- PEFT and Transformers behaviour was checked against current official sources; the experiment must
  pin the exact versions actually used.
- Reviewer confidence: 0.95 on the gate verdict; lower on whether duplicate/template clusters
  actually exist in this dataset.

## Unresolved — `USER DECISION REQUIRED`

Per `AGENTS.md` §17 these are material trade-offs the orchestration layer must not resolve alone.
All four block the gate.

- **D-1 — Improvement threshold X.** Pre-existing; `EXPERIMENT_SPEC.md` §8b has never fixed it.
  Per F16, it must be chosen from what a TED-Acc (or replacement-metric) delta *means*, not from how
  likely it is to pass.
- **D-2 — Colab tier for Phase 3–5.** Pre-existing. Determines the fp16-vs-bf16 pin, the SDPA
  backend, the image-token budget and the entire VRAM gate; must be decided before the gate is run.
- **D-3 — Duplicate-handling policy, frozen before the audit runs.** New, from F3/F4/F13. Options
  include deterministic exclusion, group-aware split construction, or halting/downgrading the
  confirmatory claim. Departing from ADR-007's official split requires a new ADR.
- **D-4 — The evaluation estimand and primary metric.** New, from F8/F9/F12. Strict verbatim
  transcription versus semantic extraction; this single choice determines the primary metric, the
  normalization rules, and the prompt's wording. To be decided after the synthetic-error probe
  (F8), which needs no model output and no test data.

## Gate status

**ADR-005 adversarial review for Phase 2 entry: BLOCK — NOT SATISFIED.**

The proposal must not be promoted into `docs/EXPERIMENT_SPEC.md` /
`docs/EVALUATION_PROTOCOL.md` / `docs/DECISIONS.md`, and `notebooks/02_baseline.ipynb` must not be
executed, until the `required` findings are resolved, the duplicate policy is frozen ahead of the
audit, the pre-registration is complete, and D-1 through D-4 are decided by the user.
