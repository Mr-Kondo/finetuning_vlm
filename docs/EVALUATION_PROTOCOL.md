# EVALUATION_PROTOCOL.md — Evaluation Protocol

**Status:** Draft — Codex adversarial review completed (Phase 0)
**Related:** [[EXPERIMENT_SPEC]] §8b (this protocol implements the "model performance improvement" success condition) / [[DECISIONS]] ADR-006, ADR-007 / [[IMPLEMENTATION_PLAN]]

This protocol implements the "model performance improvement" determination from [[EXPERIMENT_SPEC]] §8b.
The "pipeline completion" determination from [[EXPERIMENT_SPEC]] §8a is not addressed here (it is addressed in the phase gates in [[IMPLEMENTATION_PLAN]]).

## 1. Purpose

To fairly compare the Base model and the Fine-tuned model on CORD v2's held-out test split (100 samples), and to quantify the change in structured information extraction accuracy due to fine-tuning.

## 2. Conditions Compared

| Condition | Definition |
|---|---|
| Base | `Qwen/Qwen3-VL-4B-Instruct` (not fine-tuned), evaluated with a zero/few-shot prompt |
| Fine-tuned | the same base + the QLoRA adapter trained in Phase 4 applied |

Both conditions are identical in everything except the presence of the LoRA adapter, including the quantization setting (4-bit NF4) ([[DECISIONS]] ADR-006). The base model's Hub revision, processor/tokenizer/chat-template revision, quantization config (NF4, double-quant setting, compute dtype), and attention backend are pinned to the exact same commit SHA used for Fine-tuned training. Fine-tuned is this pinned artifact with the adapter applied, not a separately reloaded "equivalent" artifact ([[DECISIONS]] ADR-013). Config hashes are compared at execution time to mechanically verify there is no difference other than adapter presence.

## 3. Test Set

- The `test` split of `naver-clova-ix/cord-v2`, all 100 rows.
- Confirm before the start of Phase 5 that it has not been used for training, hyperparameter selection, or prompt tuning ([[DECISIONS]] ADR-007).
- The `validation` split (100 rows) is reserved exclusively for tuning in Phases 1–4 and is not used in this evaluation.
- Human visual inspection of the `test` images/`ground_truth` before Phase 5 is prohibited ([[DECISIONS]] ADR-008). In Phase 5, the finalized conversion logic, prompt, and parser are applied to test mechanically, exactly once.

## 4. Ensuring Fairness of the Evaluation Procedure

- Base/Fine-tuned inference goes through the same function in `src/vlm_lab/inference.py` (the only branching is whether the LoRA adapter is loaded).
- Evaluation metric computation applies the same function in `src/vlm_lab/evaluation.py` to both conditions.
- Image preprocessing (resizing, normalization) is completely identical.
- Decoding parameters (greedy or temperature=0, fixed `max_new_tokens`, fixed seed) are completely identical.
- The same prompt template used at training time is also used for Base evaluation ([[EXPERIMENT_SPEC]] §4). No prompt that disadvantages (or advantages) the Base model is used, whether intentionally or not.

## 5. Metrics

1. **JSON validity rate**: the fraction of outputs that parse as valid JSON. A foundational metric that other metrics depend on.
2. **Field-level Precision / Recall / F1**: agreement with ground truth on the flattened fields of the converted schema.
3. **Tree Edit Distance based Accuracy (TED-Acc)**: a standard evaluation metric for CORD/Donut-family tasks. Able to account for hierarchical structure and repeated fields (e.g., multiple menu-item rows). **Leading candidate for the primary metric** (finalized in [[EXPERIMENT_SPEC]] §8b).
4. **Exact Match rate**: strict match rate (reference metric).
5. **Latency / throughput**: recorded for reference only. Not included in the success conditions.

### 5.1 Normative Definition of Metrics (must be finalized in Phase 1, [[DECISIONS]] ADR-009)

"Field-level F1" and "TED-Acc" can yield different values across implementations even under the same name. The following must be defined normatively and without omission in Phase 1, appended to this document, and pre-registration completed before that. This must be finalized before observing any performance output in Phase 2.

- The type, array structure, and ordering handling of each schema field (menu-item ordering, handling of duplicate rows, item-matching method)
- Handling of missing/null values
- String/numeric normalization rules (full-width/half-width, digit grouping, currency symbols, etc.)
- The rule for extracting JSON from output, such as markdown code fences (whether/how fences are stripped)
- The F1 aggregation method: micro-F1 across samples, or macro-F1 per receipt (must be stated explicitly since it can affect the conclusion)
- For computing TED-Acc/F1, either use a pinned commit of the official Donut implementation ([Donut evaluation code](https://github.com/clovaai/donut/blob/master/donut/util.py)), or, if using a custom implementation, explicitly document the differences. The official Donut F1 is "a global micro-F1 over flattened field-value pairs," and TED-Acc is "the per-sample mean of the normalized tree edit distance" — these are not trivially the same, so any differences in a custom implementation must be recorded.
- The "s-tag format" referenced in CORD/Donut-related literature is Donut's internal tokenization representation, and is not itself the `ground_truth`/`gt_parse` JSON representation that this project stores and uses.

## 6. Statistical Treatment ([[DECISIONS]] ADR-010)

Since Base and Fine-tuned are evaluated on the same 100 test images, per-sample scores are strongly correlated (paired data). Ignoring this and comparing the confidence intervals of the two conditions separately is not an appropriate significance test for the difference ([[reviews/phase_0_adversarial.md]] Finding 4). Therefore, the following **paired bootstrap** is used.

- Compute `Δ_i = Fine-tuned_i - Base_i` (the per-sample difference in the primary metric) at the receipt level.
- From the entire test set (n=100), resample paired index sets with replacement, and compute the mean Δ for each resample; repeat this process B times (B is pre-registered; default candidate: B=10,000).
- Compute the 95% confidence interval `[CI_lower(Δ), CI_upper(Δ)]` from the resulting distribution of B mean-Δ values.
- The number of bootstrap trials B, the random seed, and the resampling method (percentile method or BCa method) are finalized in Phase 1–2 and pre-registered.
- Using the improvement threshold X finalized in [[EXPERIMENT_SPEC]] §8b, classify results into the following three categories:
  - **Improvement achieved**: `CI_lower(Δ) ≥ X`.
  - **Within margin of error (no improvement)**: `CI_lower(Δ) < X` (and the regression condition is not met).
  - **Regression**: `CI_upper(Δ) < 0`.
- The selection of the threshold X, the primary metric, and the bootstrap configuration is finalized before observing any performance output in Phase 2 (pre-registration, [[EXPERIMENT_SPEC]] §8b). Metrics or thresholds are not chosen after the fact once results are seen.

## 7. Handling of Invalid JSON Output

- Outputs that fail to parse are scored 0 for that sample in field-level F1 and TED-Acc.
- However, this zero score is separately and independently recorded as the JSON validity rate, so that the cause of a lower score on other metrics (correct structure but wrong values, versus JSON that is broken outright) is not conflated.

## 8. Artifact Format

For each evaluation run, output the following under `results/`:
- `metrics.json`: point estimates, confidence intervals, and sample counts for all metrics.
- `report.md`: a human-readable summary (Base/Fine-tuned comparison table, decision category).
- Metadata: config hash, git commit hash, seed, execution timestamp, GPU/tier information used.

## 9. What This Protocol Does Not Guarantee (Out of Scope)

- Generalization performance to datasets other than CORD.
- Comparison against other model sizes or other training methods.
- Whether production-scale latency/throughput requirements are met (latency is recorded for reference only).
