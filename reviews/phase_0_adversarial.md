# Phase 0 Adversarial Review

- **Review date:** 2026-08-10
- **Reviewed phase:** Phase 0 — Design/specification drafting (no implementation code exists yet)
- **Reviewer:** Codex (via `codex-companion.mjs`, job `task-msn32ovi-u1844j`, read-only task, confidence 0.94)
- **Reviewed specification / design (files read in full):**
  - `docs/EXPERIMENT_SPEC.md`
  - `docs/EVALUATION_PROTOCOL.md`
  - `docs/IMPLEMENTATION_PLAN.md`
  - `docs/DECISIONS.md`
  - `docs/STATE.md`
  - `CLAUDE.md`
  - `AGENTS.md`
- **Review outcome:** **Design revision required. Do not proceed to Phase 1 as-is.**

---

## 1. Context (Codex's own framing)

Codex read all seven documents in full and cross-checked model/dataset/CORD-Donut evaluation implementation/Qwen3-VL architecture/T4 GPU constraints against primary sources. No files were changed, no code was executed, no dataset was downloaded.

Facts Codex confirmed independently:
- `Qwen/Qwen3-VL-4B-Instruct` is a 4B, image-text-to-text, Apache-2.0 model.
- `naver-clova-ix/cord-v2` has splits train=800 / validation=100 / test=100, columns `image` and `ground_truth`.
- `ground_truth` is a JSON string; the extraction target within it is `gt_parse`.

Explicitly **not verified** by Codex: actual Colab VRAM behavior, cross-split duplication, exact LoRA target module names, and whether CORD appeared in Qwen's pretraining corpus.

---

## 2. Findings (ranked most severe first)

### 1. Phase 2 consumes the held-out test set before Phase 5 — [VERIFIED]
- **Files:** `IMPLEMENTATION_PLAN.md` Phase 2 section (line 62) states Phase 2 runs the Base model on "held-out test (or a small in-phase subset)". This directly contradicts `DECISIONS.md` ADR-007 (line 128, "test is used exactly once, in Phase 5") and `EXPERIMENT_SPEC.md` §3 (line 45, "used only for the final Phase 5 comparison").
- **Failure scenario:** After seeing 10 test predictions/baseline numbers in Phase 2, the prompt, `max_new_tokens`, image resolution, schema, or threshold X gets tuned to fix cases where JSON breaks. By the time Phase 5 re-runs, test is no longer held-out.
- **Fix:** Restrict Phase 2's smoke check to a fixed train/validation subset only. No test image, ground truth, prediction, or aggregate metric may be generated or displayed until the single, fully-frozen Phase 5 run. Also remove the test reference currently in `EXPERIMENT_SPEC.md` §8a-3 (line 84).

### 2. Pre-registration happens too late and is incomplete — [VERIFIED]
- **Files:** `EXPERIMENT_SPEC.md` §8b (line 94) says the primary metric and X are fixed during "Phase 1–3, before training runs". `EVALUATION_PROTOCOL.md` §6 (line 45) says the same. But under the current plan, Phase 2 can already see test baseline numbers. Base is defined only as "zero/few-shot" — shot count, demo source, and demo IDs are undefined.
- **Failure scenario:** After seeing Base's test results, pick whichever metric maximizes the apparent gap and set an X that is known to be achievable — technically compliant with "before training" but not with the spirit of pre-registration.
- **Fix:** Freeze the following before any performance output is observed:
  - full formula for the primary metric, X, and the decision rule
  - zero-shot vs. few-shot; if few-shot, fixed demo IDs drawn from train
  - prompt-selection procedure
  - parser, normalization, and aggregation method
  - hyperparameter search space and trial budget
  - checkpoint selection, early stopping, tie-breaking rule
  - number of allowed test executions and re-run conditions

  Choosing values *using validation* is fine; the *selection procedure itself* must be fixed before selection happens.

### 3. Phase 1 lets a human see test ground truth — [VERIFIED]
- **Files:** `IMPLEMENTATION_PLAN.md` Phase 1 (line 57) requires "verifying conversion correctness across all splits including test"; `EXPERIMENT_SPEC.md` §8a-2 (line 84) requires the same for schema validation.
- **Failure scenario:** By visually inspecting test conversion output, unknown test-specific fields, nested menu structures, or missing values become known before the conversion rule or prompt is finalized. This leaks into evaluation design even if test is never used for training.
- **Fix:** Do conversion-logic development and visual inspection on train/validation only. Apply the frozen converter to test mechanically in Phase 5, with only pre-defined machine checks (record count, parse success/failure). If a test-specific failure is observed and fixed, downgrade that test evaluation run to exploratory (non-confirmatory).

### 4. The statistical decision rule is not a paired comparison — [VERIFIED]
- **File:** `EVALUATION_PROTOCOL.md` §6 (line 45) defines success as "the two 95% confidence intervals do not overlap".
- **Failure scenario:** Base and Fine-tuned are evaluated on the same 100 images, so per-sample scores are strongly correlated. Comparing separate CIs discards this pairing and is not equivalent to a significance test on the difference. A point-estimate gap ≥ X can coexist with a difference-CI that widely includes values below X.
- **Fix:** Use paired bootstrap resampling at the receipt level to estimate the per-sample difference Δ = Fine-tuned − Base directly. Pre-register the bootstrap method, iteration count, seed, and CI construction. If claiming "statistically supported improvement of at least X," require `CI_lower(Δ) ≥ X`; if a looser rule (point estimate ≥ X, statistically > 0) is intended, state that explicitly.

### 5. Pipeline completion depends on model output quality — [VERIFIED]
- **Files:** `EXPERIMENT_SPEC.md` §8a-3 (line 84) requires Base to "return parseable output" as a pipeline-completion condition, while `EVALUATION_PROTOCOL.md` §7 (line 54) legitimately scores invalid JSON as 0.
- **Failure scenario:** A correctly functioning Base breaks JSON on 1 out of 100 samples. Inference, storage, invalid-detection, and zero-scoring all worked correctly, yet pipeline completion is marked as failed and blocks the scientific evaluation. Conversely, a broken pipeline that always emits `{}` would pass, since `{}` parses.
- **Fix:** Redefine pipeline completion as "raw output was obtained for every sample, the parser classified each as valid/invalid, and both valid and invalid cases are recorded in evaluation artifacts." Move JSON validity and scoring entirely into the model-performance track. Technical sanity checks should be based on record counts, sample-ID alignment, non-missing raw output, exception logging, and artifact integrity.

### 6. Ground-truth schema and metrics are underspecified enough to be gamed — [VERIFIED]
- **Files:** `EXPERIMENT_SPEC.md` §3–4 (line 32) only says conversion targets "a clean JSON schema"; `EVALUATION_PROTOCOL.md` §5 (line 37) names field-level F1 and TED-Acc without defining them.
- **Failure scenario:**
  - Dropping difficult fields during conversion artificially inflates both models' scores.
  - Menu array ordering, duplicates, and item alignment could be handled inconsistently between implementers.
  - Micro-F1 vs. macro-F1 can flip the conclusion.
  - Differences in markdown-fence stripping, numeric normalization, or empty/null handling could rescue one model's output but not the other's.
- **Fix:** Normatively define every schema field, type, array structure, ordering, missing/null handling, duplicate handling, string/numeric normalization, JSON-fence stripping, and Exact Match canonicalization. Either use a pinned commit of the official Donut implementation for TED/F1 or explicitly document any deviation. Note: the official Donut F1 is a global micro-F1 over flattened field-value pairs, and TED-Acc is the per-sample mean of normalized tree edit distance — these are not simply "the same-named metric" by default. Also, the "s-tag format" referenced in some CORD/Donut discussions is an internal Donut tokenization concept, not the stored `ground_truth`/`gt_parse` JSON itself.

### 7. Split-name separation alone does not prevent content-duplication leakage — [HYPOTHESIS]
- **File:** `DECISIONS.md` ADR-007 (line 128) only fixes split *usage*; no exact/near-duplicate audit of images, receipts, or store templates is addressed anywhere.
- **Failure scenario:** Identical or near-identical receipts exist in both train and test, and the fine-tuned model memorizes content or templates. With only 100 test samples, even a handful of duplicates could meaningfully inflate the apparent improvement.
- **Fix:** Before training, run an automated cross-split audit using image hashes/perceptual hashes and canonical hashes of the ground-truth structure. Pre-register the duplicate threshold and the policy for handling matches found (exclusion, group-aware re-split, reporting). Note: actual duplication has not been confirmed to exist.

### 8. What exactly LoRA trains is undecided — [VERIFIED]
- **File:** `EXPERIMENT_SPEC.md` §5 (line 56) lists candidate module names like `q_proj` but does not specify whether the target is the language tower, the vision encoder, or the projector. Whether the training loss is masked to assistant JSON tokens only is also not addressed.
- **Failure scenario:** Selecting modules by suffix name alone could unintentionally attach LoRA to vision-side projections sharing the same name, while a different implementation might target only the language side. This changes VRAM usage, trainable parameter count, and the experiment's actual meaning. Qwen3-VL has independent text/vision configs, confirmed from the official config.
- **Fix:** Turn the Phase 1 checklist item into an approval gate requiring fully-qualified module names, the specific tower targeted, adapter parameter count, vision/projector freeze status, assistant-only label masking, and a list of trainable parameters — not just a name check. Record the decision as an ADR before training implementation proceeds.

### 9. "Same 4-bit" alone does not guarantee fairness or reproducibility — [VERIFIED]
- **Files:** `EVALUATION_PROTOCOL.md` §2–4 (line 14) states everything except the LoRA adapter is identical between Base and Fine-tuned, but `EXPERIMENT_SPEC.md` §2 (line 25) leaves the choice between runtime quantization and a pre-quantized (e.g. Unsloth) artifact undecided.
- **Failure scenario:** The Fine-tuned adapter is trained against a specific runtime-quantized revision, while Base evaluation loads a different pre-quantized repo. Both are "4-bit NF4" nominally, but quant values, double-quant setting, compute dtype, or processor/chat template differ — so the measured difference is no longer isolated to the LoRA adapter.
- **Fix:** Pin the identical base revision, processor/tokenizer/chat-template revision, quantization path, NF4 config, double-quant setting, compute dtype, attention backend, eval mode, and generation config across both conditions. Fine-tuned must apply its adapter to that exact base artifact. Add a runtime check that config hashes match except for adapter presence.

### 10. VRAM gating and smoke conditions are not representative of production load — [HYPOTHESIS]
- **Files:** `EXPERIMENT_SPEC.md` §6/§10 (line 64) leaves Colab tier, image token budget, and LoRA rank undecided; `IMPLEMENTATION_PLAN.md` §3 (line 82) defines smoke/mini/full with different values, where smoke uses a tiny dataset and only a few steps.
- **Failure scenario:** A smoke run with short receipts, low resolution, and small rank passes on T4, but the full run OOMs at max image-token count, longer JSON targets, a different rank, or after optimizer state is allocated. Reducing record count alone does not validate per-batch activation peak memory.
- **Confirmed external constraints:** Qwen3-VL-4B is distributed in BF16. The standard FlashAttention-2 CUDA kernels target Ampere/Ada/Hopper; T4 (Turing) requires a separate code path. NVIDIA's official T4 precision table lists FP32/FP16/INT8/INT4.
- **Fix:** Define a quantitative go/no-go gate at the Phase 1 exit: measure p50/p95/max image-token and target-token counts, use the production-intended rank/target modules, and measure forward/backward peak VRAM after optimizer state is allocated. Smoke should reduce record count and step count only — image size, sequence length, rank, and microbatch size should match full-scale worst case. Design an explicit T4 fallback path (e.g. FP16/SDPA). Whether 4B fits in 16GB cannot be asserted before this is actually measured.

### 11. External artifact revisions are not pinned — [VERIFIED]
- **Files:** `EXPERIMENT_SPEC.md` §7 (line 71) and `EVALUATION_PROTOCOL.md` §8 (line 59) record config hash, project commit, and package versions, but model/dataset/processor Hub revisions and dataset fingerprint are not addressed.
- **Failure scenario:** The Hub `main` branch is updated later; even with the same repo ID, weights, chat template, preprocessing, or dataset parquet files change. A later reproduction run becomes a different experiment without anyone noticing.
- **Fix:** Pin model, processor, dataset, and quantization repos to specific commit SHAs; record dataset fingerprint, split counts, and content hashes where possible in run artifacts. (Note: the project directory currently not being a git repo is not part of this finding.)

### 12. A single training seed cannot characterize training randomness — [HYPOTHESIS]
- **Files:** `EXPERIMENT_SPEC.md` §7 (line 71) fixes a single seed; `EVALUATION_PROTOCOL.md` §6 (line 45) only bootstraps over the 100 test samples.
- **Failure scenario:** Due to LoRA initialization or data ordering, seed A shows +X improvement while seeds B/C show none. The tight receipt-level bootstrap CI only characterizes uncertainty conditional on that one fixed training run, not general reproducibility of the improvement.
- **Fix:** If GPU budget allows, train with multiple pre-registered seeds and report cross-seed variance. If budget forces a single seed, explicitly scope the conclusion as "a conditional comparison for this fixed seed/checkpoint," not a general, reproducible improvement claim.

---

## 3. Counterarguments (Codex's own self-check)

- ADR-006 (same inference function, same prompt, same quantization) is a reasonable foundation, but "same function" as a structural constraint does not by itself guarantee identical artifact revisions or identical effective runtime configuration.
- If Phase 1's test schema validation were fully automated, never shown to a human, and the spec were never changed afterward, it would not constitute leakage — but the current documents do not impose that constraint.
- Whether public CORD was included in Qwen's pretraining data is unverified. Even if it were, since Base and Fine-tuned share the same base model, this would not directly invalidate the paired difference attributable to adding the adapter — but it would affect how "generalization to unseen receipts" is interpreted and could create a ceiling effect. This should be explicitly noted in Threats to Validity.
- Multiple seeds multiply GPU cost several-fold; not strictly required if a single-run, scope-limited conclusion is accepted.

## 4. Key Points (blocking changes before Phase 1, priority order)

| Priority | Blocking change before Phase 1 | Reason |
|---:|---|---|
| 1 | Fully exclude test from Phase 2 | Current design breaks held-out status |
| 2 | Fix pre-registration content and freeze timing | Prevents post-hoc metric/threshold/prompt selection |
| 3 | Make test conversion blind | Prevents design changes informed by test ground truth |
| 4 | Switch to paired-difference CI | Current CI-overlap rule doesn't fit the paired comparison |
| 5 | Normatively define schema/evaluator | Same-named metrics can yield different results |
| 6 | Separate pipeline completion from JSON quality | Resolves conflation of technical success and model performance |

## 5. Examples (illustrative, from Codex)

- Whether Base output wrapped in a markdown code fence counts as invalid (or valid after fence-stripping) can swing scores by several points depending on parser behavior — this must be specified.
- For a receipt with a 10-line menu, micro-F1 is dominated by menu items; receipt-level macro-F1 weights short and long receipts equally. Both are legitimately called "field-level F1," so the exact definition must be pre-specified.

## 6. Plain-language summary (Codex's own framing)

The current design has largely worked out "how to run the pipeline end-to-end," but has not yet finished "how to seal the scoring method and the exam questions before looking at the answers."

## 7. Self-critique (Codex's own caveats)

- Cross-split duplication, actual Colab VRAM behavior, exact LoRA module names, cross-seed variance, and whether CORD leaked into Qwen's pretraining were not investigated or measured — hence tagged [HYPOTHESIS].
- The HF dataset config name is literally `cord-v2`; if the docs' phrase "default config" means "the config that gets selected by default," that's not a critical error, but it would be inaccurate if used literally as a `"default"` string in code.
- Stated confidence: **0.94** — high confidence on document-to-document contradictions, statistical methodology, pipeline-condition ambiguity, and evaluation-spec gaps; medium confidence on actual VRAM behavior, duplication, and training variance (unmeasured).

## 8. References (as supplied by Codex)

- [Qwen3-VL-4B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [Qwen3-VL official Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [CORD v2 dataset metadata](https://huggingface.co/datasets/naver-clova-ix/cord-v2/blob/main/dataset_infos.json)
- [CORD official repository](https://github.com/clovaai/cord)
- [Donut data format](https://github.com/clovaai/donut/blob/master/README.md)
- [Donut official JSON evaluator](https://github.com/clovaai/donut/blob/master/donut/util.py)
- [Donut official test aggregation](https://github.com/clovaai/donut/blob/master/test.py)
- [bitsandbytes 4-bit/QLoRA explanation](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [FlashAttention official repository](https://github.com/Dao-AILab/flash-attention)
- [NVIDIA T4 datasheet](https://www.nvidia.com/en-us/data-center/tesla-t4/)

---

## 9. Verdict

**Do not proceed to Phase 1 as-is. Design revision is required.**

Minimum checklist to complete before proceeding:

- [ ] Remove all test references from Phase 2 and the pipeline smoke check; seal test to a single Phase 5 run
- [ ] Make test conversion and schema validation blind (no human inspection of test ground truth before Phase 5)
- [ ] Pre-register primary metric, X, paired-bootstrap decision rule, prompt/shot design, parser, normalization, and checkpoint selection — all before any performance output is seen
- [ ] Fully define the converted schema's deviations from official Donut conventions
- [ ] Remove "model returns valid JSON" from the pipeline-completion definition
- [ ] Document the exact LoRA target tower(s), label masking, and precise base/quantization artifact
- [ ] Define a production-shape VRAM go/no-go gate before the smoke test
- [ ] Pin model/dataset/processor Hub revisions

Once these are addressed, the design would be adequate to proceed into Phase 1 environment setup and data preparation.

---

## 10. Unresolved questions

- Actual cross-split (train/test) content duplication in CORD v2 has not been measured — requires an image-hash/perceptual-hash audit during Phase 1.
- Actual Colab VRAM behavior for the full-scale QLoRA config (max image tokens, target rank, microbatch) has not been measured — requires a production-shape smoke gate before Phase 3/4.
- Exact fully-qualified LoRA target module names for Qwen3-VL-4B-Instruct (language tower vs. vision/projector) have not been confirmed against the live model architecture.
- Whether CORD (or CORD-derived data) appeared in Qwen's pretraining corpus is unknown; its effect on interpreting "generalization" needs to be added to Threats to Validity regardless of the answer.
- Whether the project will afford multi-seed training runs, or accept a single-seed, scope-limited conclusion, is a cost/rigor tradeoff for the user to decide.

---

## 11. Claude Code Disposition (2026-08-10)

Processed per `CLAUDE.md` §6 / `AGENTS.md` §10–12. All 12 material findings were read in full and cross-checked against `EXPERIMENT_SPEC.md`, `EVALUATION_PROTOCOL.md`, `IMPLEMENTATION_PLAN.md`, and `DECISIONS.md` before disposition. No finding was dismissed without a documented rationale.

| # | Finding | Disposition | New/Amended ADR | Rationale |
|---:|---|---|---|---|
| 1 | Phase 2 consumes test before Phase 5 | **ACCEPT** | ADR-008 | Direct, verified contradiction between `IMPLEMENTATION_PLAN.md` Phase 2 and ADR-007/`EXPERIMENT_SPEC.md` §3. No credible counter-rationale exists; this is a held-out integrity defect, not a design preference. |
| 2 | Pre-registration too late / incomplete | **ACCEPT** | ADR-009 | "Before training" leaves a window where Phase 2 baseline output could inform metric/threshold choice even without touching test. Codex's distinction ("choosing values from validation is fine; the selection *procedure* must be fixed first") is methodologically correct and cheap to adopt now, before any code exists. |
| 3 | Phase 1 lets a human see test ground truth | **ACCEPT** | ADR-008 | Same root cause as #1 (test exposure before Phase 5). Blind mechanical processing of test in Phase 1/5 is a low-cost structural fix. |
| 4 | CI-overlap is not a paired comparison | **ACCEPT** | ADR-010 | Statistically correct: Base/Fine-tuned share the same 100 images, so scores are correlated. Paired bootstrap on the per-sample difference is the standard fix and does not add meaningful implementation cost. |
| 5 | Pipeline completion depends on model output quality | **ACCEPT** | ADR-011 | Directly undermines the ADR-004 pipeline/performance separation this project is built around (a single bad JSON sample from an otherwise-correct Base run would wrongly fail "pipeline completion"; a broken pipeline emitting `{}` would wrongly pass). Redefinition is a clarification of existing intent, not a new experiment design. |
| 6 | Schema/metric definitions underspecified | **ACCEPT** | ADR-009 | Field ordering, null handling, micro vs. macro F1, and fence-stripping can each swing scores by multiple points on a 100-sample test set. Folded into the same pre-registration ADR as #2 since both concern "freeze the measurement procedure before looking at results." Full field-by-field schema definition itself is deferred to Phase 1 execution (requires inspecting train/validation data), but the *requirement* to define it before Phase 2 is accepted now. |
| 7 | Split-name separation doesn't prevent content-duplication leakage | **ACCEPT** (tagged HYPOTHESIS by Codex — actual duplication unconfirmed) | ADR-008 | Even though duplication is unconfirmed, the audit is cheap (automated hashing, no design cost) relative to the risk on a 100-sample test set. Accepted as a Phase 1 task; not blocking on itself since it produces evidence rather than requiring a redesign. |
| 8 | What exactly LoRA trains is undecided | **ACCEPT** | ADR-012 | Qwen3-VL has independent text/vision configs (confirmed by Codex against the official config), so suffix-only module matching is a real risk, not speculative. Elevating this from a checklist item to an approval gate costs nothing extra since Phase 1 already had to determine this. |
| 9 | "Same 4-bit" doesn't guarantee fairness/reproducibility | **ACCEPT** | ADR-013 | Directly threatens ADR-006's fairness guarantee (isolating the measured difference to the LoRA adapter). Revision pinning and a config-hash consistency check are standard reproducibility practice. |
| 10 | VRAM gating/smoke conditions not representative of production load | **ACCEPT** (tagged HYPOTHESIS by Codex) | ADR-014 | Even unconfirmed, a reduced-shape smoke test provably cannot detect a production-shape OOM — this is a logical gap in the current plan, not a probabilistic risk that needs prior evidence. Confirmed external constraints (T4/FlashAttention-2 incompatibility, BF16 distribution) make the risk concrete enough to act on now. |
| 11 | External artifact revisions not pinned | **ACCEPT** | ADR-015 | Standard reproducibility requirement; no counter-rationale considered. |
| 12 | Single training seed cannot characterize training randomness | **USER DECISION REQUIRED** | none yet | Codex itself frames this as a cost/rigor tradeoff for the user (GPU budget vs. cross-seed variance characterization). Per `AGENTS.md` §11, acceptance-threshold/rigor tradeoffs of this kind are not Claude Code's to resolve unilaterally. **Does not block Phase 1**: it only affects Phase 4 training design and the scope of the final claim. Interim default adopted without user input: if only one seed is run, the final conclusion is explicitly scoped as "a conditional comparison for this fixed seed/checkpoint" rather than a general reproducibility claim (`EXPERIMENT_SPEC.md` §8b, §9). The user will be asked separately whether to budget for multi-seed training. |

**Non-finding fix:** Codex's self-critique noted the CORD v2 HF config name is literally `cord-v2`, not `default`; `EXPERIMENT_SPEC.md` §3 was corrected to avoid implying `"default"` is a literal string to use in code. Not treated as a material finding requiring an ADR.

**Counterarguments incorporated:** The pretraining-data-contamination caveat (CORD possibly present in Qwen's pretraining corpus) was added to `EXPERIMENT_SPEC.md` §9 Threats to Validity per Codex's own self-check in §3, even though it does not invalidate the paired-difference comparison.

**Gate status:** All ACCEPT findings have been reflected in `EXPERIMENT_SPEC.md`, `EVALUATION_PROTOCOL.md`, `IMPLEMENTATION_PLAN.md`, and `DECISIONS.md` (ADR-008 through ADR-015). No finding was rejected or deferred outright. One finding (#12) is escalated to the user and does not block Phase 1. `docs/STATE.md` updated accordingly. The ADR-005 gate is satisfied pending user sign-off on this disposition and on the seed-strategy question.
