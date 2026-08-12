# DECISIONS.md — Decision Log (ADR)

This file records the important decisions of this project in Architecture Decision Record (ADR) format.
**Past ADRs are not edited.** To reverse a decision, add a new ADR and change the old ADR's Status to `Superseded by ADR-XXX`.

---

## ADR-001: Adopt Qwen3-VL-4B-Instruct as the Primary Model

- **Date:** 2026-08-10
- **Status:** Accepted

**Context:**
Running a QLoRA fine-tuning experiment that completes entirely on Colab requires (a) a size that fits within the GPU VRAM of the free/Pro tier, (b) an instruction-tuned VLM that actually exists and is publicly available, and (c) a model that can be handled straightforwardly with the HF `transformers` / `peft` / `bitsandbytes` stack.

**Decision:**
Adopt `Qwen/Qwen3-VL-4B-Instruct` as the primary model. Its existence has been confirmed on the HF Hub (as of 2026-08-10: downloads 3.6M, license apache-2.0, task: image-text-to-text, library: transformers). The existence of the pre-quantized 4-bit version `unsloth/Qwen3-VL-4B-Instruct-bnb-4bit` has also been confirmed and will be considered as a loading option in Phase 1.

**Alternatives Considered:**
- Qwen3-VL-8B/32B-Instruct: rejected because they are highly unlikely to fit in the Colab free tier (T4 16GB) even with 4-bit QLoRA.
- Qwen3-VL-2B-Instruct: safest in terms of VRAM, but 4B was chosen to prioritize headroom for measurable performance improvement from fine-tuning on the structured-extraction task.

**Consequences:**
- Whether it actually runs on Colab T4 (16GB) needs to be measured in Phase 1 (EXPERIMENT_SPEC.md §Open Verification Items).
- If a change in model size becomes necessary, add an ADR.

---

## ADR-002: Adopt CORD v2 as the Primary Dataset

- **Date:** 2026-08-10
- **Status:** Accepted

**Context:**
To choose a task that makes the effect of VLM fine-tuning easy to measure, we select image → structured JSON extraction (document understanding). A dataset with a size that allows training/evaluation to run quickly and with clear train/validation/test splits is needed.

**Decision:**
Adopt `naver-clova-ix/cord-v2` as the primary dataset. Its structure has been confirmed on the HF Hub (as of 2026-08-10): `default` config, splits = train 800 / validation 100 / test 100, columns = `image` (Image), `ground_truth` (string, a Donut-format JSON string containing `gt_parse`).

**Alternatives Considered:**
- `mychen76/receipt_cord_ocr_v2`: found via search, but rejected due to low reliability (sparse dataset-card information: downloads 127, "More Information needed").
- SROIE and other receipt datasets: structure and license on HF unconfirmed, so out of scope for now (held as a candidate for future generalization experiments).

**Consequences:**
- `ground_truth` is a Donut-format nested JSON string, requiring conversion into the input/output format used for instruction tuning ([[EXPERIMENT_SPEC]] §3, the responsibility of `data.py`).
- The small size of the test split (100 rows) leads to high variance in evaluation metrics, which is addressed in [[EVALUATION_PROTOCOL]].

---

## ADR-003: Adopt 4-bit QLoRA as the Training Method

- **Date:** 2026-08-10
- **Status:** Accepted

**Context:**
Full fine-tuning of the entire model on Colab (especially the free tier T4 16GB) is not realistic.

**Decision:**
Adopt QLoRA: 4-bit NF4 quantization (bitsandbytes) + a LoRA adapter (PEFT). Base model weights are frozen, and only the LoRA adapter in the attention/MLP projection layers is trained.

**Alternatives Considered:**
- Full fine-tuning: rejected due to insufficient VRAM.
- 8-bit LoRA: has more VRAM headroom than 4-bit, but validating Colab compatibility first with the 4B model + QLoRA was judged higher priority. Retained as a candidate for a future comparison experiment.

**Consequences:**
- The effect of quantization noise on evaluation metrics is documented as a threat to validity in [[EXPERIMENT_SPEC]].
- LoRA target modules / rank / alpha are undecided and will be finalized in Phase 1–3, with an ADR added.

---

## ADR-004: Define Pipeline Completion and Model Performance Improvement as Independent Success Conditions

- **Date:** 2026-08-10
- **Status:** Accepted

**Context:**
An explicit requirement from the user. In VLM fine-tuning experiments, "the code ran to completion" and "fine-tuning improved model performance" are easily conflated. This conflation invites two kinds of failure mode: (a) wasting time hunting for implementation bugs when performance fails to improve, and (b) conversely, dismissing a case where a bug caused performance to degrade as merely a "negative result."

**Decision:**
Define the two success conditions as completely separate (detailed in [[EXPERIMENT_SPEC]] §8):
1. **Pipeline completion**: everything from environment setup through the smoke test, full training, running evaluation on both Base and Fine-tuned, and saving artifacts completes to the end without crashing. A binary engineering-achievement condition unrelated to whether the numeric results are good or bad.
2. **Model performance improvement**: on the held-out test set, the fine-tuned model exceeds the base model by at least the pre-registered threshold. A scientific experimental outcome; failure (i.e., no improvement) is also reported as a legitimate result.

**Consequences:**
- If pipeline completion is not achieved, evaluating model performance improvement becomes meaningless on its own, so this is a blocking condition.
- Even if model performance improvement is not achieved, that alone is not a project failure (it is recorded as a negative result in [[STATE]] / the final report).
- During implementation, these two conditions must not be mixed into the same checklist or log.

---

## ADR-005: Require Codex Adversarial Review as a Mandatory Gate Before Implementation

- **Date:** 2026-08-10
- **Status:** Accepted

**Context:**
An explicit requirement from the user. Since each phase transition on Colab costs GPU time and real time, design-stage flaws (data leakage, evaluation unfairness, undefined thresholds, etc.) should be eliminated before implementation.

**Decision:**
At the time the Phase 0 design specification is finalized, and before any subsequent "phase transition involving implementation," a Codex adversarial review is performed; findings are addressed and user approval is obtained before starting the next phase (also documented as an operating rule in [[CLAUDE]] / [[AGENTS]]). Minor documentation-only updates are outside the scope of this gate.

**Alternatives Considered:**
- Proceeding to implementation without review: rejected due to the high risk of wasting Colab execution cost.

**Consequences:**
- An adversarial review is performed on this full set of docs before the Phase 0 → Phase 1 transition (planned within this session).
- Review results are stored under `reviews/` ([[IMPLEMENTATION_PLAN]]).

---

## ADR-006: Base/Fine-tuned Evaluation Uses the Same Code Path and the Same Decoding Parameters

- **Date:** 2026-08-10
- **Status:** Accepted

**Context:**
Using separate inference code or prompts for evaluating the Base model and the Fine-tuned model would compromise the fairness of the comparison, risking over- or under-estimating the effect of fine-tuning.

**Decision:**
Fully share `src/vlm_lab/inference.py` and `evaluation.py` between Base and Fine-tuned, using identical prompt templates, image preprocessing, and decoding parameters (greedy or temperature=0, fixed max_new_tokens, fixed seed). The only difference is the presence or absence of the LoRA adapter.

**Consequences:**
- Structurally prevents staging "the effect of fine-tuning" by using a prompt that disadvantages the Base model.
- The prompt template is decided in Phase 1, and once decided, is pinned and recorded in configs/.

---

## ADR-007: The Held-Out Test Split Is Never Used for Hyperparameter Selection

- **Date:** 2026-08-10
- **Status:** Accepted

**Context:**
The `test` split (100 rows) of cord-v2 should be reserved exclusively for the final comparison; using it for model selection during training, early stopping, or prompt tuning would cause leakage and undermine the reliability of the reported performance improvement.

**Decision:**
Use only the `validation` split (100 rows) for hyperparameter search, prompt tuning, and checkpoint selection. The `test` split is used exactly once, in the final Phase 5 evaluation.

**Consequences:**
- The smoke/mini iterations in Phase 3/4 reference only part of `validation` and do not touch `test` (confirmed at the phase gates in [[IMPLEMENTATION_PLAN]]).

---

## ADR-008: Add a Test-Split Blindness Principle and a Cross-Split Duplication Audit

- **Date:** 2026-08-10
- **Status:** Accepted
- **Trigger:** `reviews/phase_0_adversarial.md` Findings 1, 3, 7 (Codex adversarial review, Phase 0). ACCEPT.

**Context:**
Codex's adversarial review identified the following design defects.
1. `IMPLEMENTATION_PLAN.md` Phase 2 described running baseline inference against "the held-out test (or a small in-phase subset)," which directly contradicted `EXPERIMENT_SPEC.md` §3 and ADR-007's rule that "test is used only in Phase 5." Seeing test predictions or baseline numbers in Phase 2 could let that knowledge leak into subsequent prompt, threshold, or schema adjustments.
2. `IMPLEMENTATION_PLAN.md`'s Phase 1 data-conversion verification, and `EXPERIMENT_SPEC.md` §8a-2, stated that conversion correctness would be "confirmed across all splits including test," which permitted a human to visually inspect test's ground truth during development. Knowledge of test's content could influence decisions about the conversion logic or schema definition.
3. Split-name separation alone cannot detect leakage from substantial duplication (near-identical) of images or receipt templates between train and test. Whether such duplication actually exists is unconfirmed.

**Decision:**
1. Baseline inference in Phase 2 targets only a small subset of `train`/`validation`; the `test` images, ground truth, predictions, and aggregate metrics are not generated or displayed at all in Phase 2.
2. Development, debugging, and visual inspection of the `data.py` conversion logic are done using `train`/`validation` only. Once the conversion logic is finalized, it is applied to `test` mechanically exactly once, during Phase 1 or Phase 5, with only automated checks such as record count and parse success/failure. No human visual inspection is performed.
3. In Phase 1, perform an automated cross-split duplication audit between train/test using image hashes/perceptual hashes and normalized hashes of the ground-truth structure. The duplicate threshold and the handling policy (exclusion / group-aware re-split / reporting only) are pre-registered before the audit is run.

**Alternatives Considered:**
- Evaluating the full test set for connectivity checking in Phase 2 (the prior plan): rejected, as Codex correctly noted this compromises held-out status.
- Omitting the duplication audit: rejected, since on a small 100-sample test set, even a handful of duplicates could meaningfully distort the measured improvement, and the automated audit is low-cost.

**Consequences:**
- `EXPERIMENT_SPEC.md` §3, §8a-3, §9, §10 / `EVALUATION_PROTOCOL.md` §3 / `IMPLEMENTATION_PLAN.md` Phase 1 and Phase 2 were updated to align with this ADR.
- This does not discard ADR-007 (the restriction on test usage); it concretizes and strengthens its operation.

---

## ADR-009: Tighten Pre-Registration Content and Timing, and Normatively Define the Evaluation Metrics/Schema

- **Date:** 2026-08-10
- **Status:** Accepted
- **Trigger:** `reviews/phase_0_adversarial.md` Findings 2, 6 (Codex adversarial review, Phase 0). ACCEPT.

**Context:**
Codex's findings: (a) `EXPERIMENT_SPEC.md` §8b and `EVALUATION_PROTOCOL.md` §6 defined the pre-registration timing only as "before training is executed," which technically permitted choosing the primary metric or threshold X after seeing Phase 2's test baseline numbers (contrary to the spirit of pre-registration even if technically compliant). Also, whether Base is zero-shot or few-shot, the shot count, and the demo-selection procedure were undefined. (b) "Field-level F1" and "TED-Acc" can differ in value across implementations under the same name, but field types, array ordering, missing-value handling, normalization rules, and the choice of micro vs. macro aggregation were undefined.

**Decision:**
- Move the pre-registration baseline from "before training is executed" up to "before observing any performance output in Phase 2."
- Specify, as required pre-registration elements, the primary metric formula, X, and the decision rule, plus the shot design, demo-selection procedure, prompt-selection procedure, parser normalization specification, hyperparameter search space, checkpoint-selection rule, and allowed number of test executions.
- Add a normative definition of the schema and metrics (§5.1) to `EVALUATION_PROTOCOL.md`, requiring either a reference to a pinned commit of the official Donut implementation or an explicit statement of differences for a custom implementation.

**Alternatives Considered:**
- Keeping the current "before training is executed" baseline: even combined with preventing test exposure in Phase 2 (addressed by ADR-008), room would remain to choose metrics after seeing validation-based baseline output, so a stricter baseline was adopted.

**Consequences:**
- `EXPERIMENT_SPEC.md` §8b, §10 / `EVALUATION_PROTOCOL.md` §5, §6 / `IMPLEMENTATION_PLAN.md` Phase 1 exit conditions were updated to align with this ADR.
- Having the full pre-registration set finalized by the end of Phase 1 becomes a condition for starting Phase 2.

---

## ADR-010: Change the Base/Fine-tuned Comparison Decision Rule to a Paired-Bootstrap Difference Confidence Interval

- **Date:** 2026-08-10
- **Status:** Accepted
- **Trigger:** `reviews/phase_0_adversarial.md` Finding 4 (Codex adversarial review, Phase 0). ACCEPT.

**Context:**
`EVALUATION_PROTOCOL.md` §6 (prior version) used "the 95% confidence intervals of Base and Fine-tuned do not overlap" as the improvement-decision condition. However, since Base and Fine-tuned are evaluated on the same 100 test images, per-sample scores are strongly correlated (paired data). Deciding based on the overlap of independent confidence intervals discards this pairing and is not an appropriate significance test for the difference. Even if the point-estimate gap is at least X, the confidence interval of the difference can include values well below X.

**Decision:**
Change the decision method to a confidence interval `CI(Δ)` from paired bootstrap (resampling with pairing) on the receipt-level difference `Δ_i = Fine-tuned_i - Base_i`.
- Improvement achieved: `CI_lower(Δ) ≥ X`
- Within margin of error: `CI_lower(Δ) < X` and the regression condition is not met
- Regression: `CI_upper(Δ) < 0`
The number of bootstrap trials, the seed, and whether to use the percentile method or the BCa method are finalized in Phase 1–2 and pre-registered (ADR-009).

**Alternatives Considered:**
- Deciding via the overlap of independent confidence intervals (the prior plan): rejected, as it ignores the correlation in paired data and carries both false-negative and false-positive risk.
- Deciding solely on the point-estimate difference: rejected, as it removes the quantification of uncertainty from the confidence interval.

**Consequences:**
- `EVALUATION_PROTOCOL.md` §6 / `EXPERIMENT_SPEC.md` §8b were updated to align with this ADR.

---

## ADR-011: Decouple the Definition of Pipeline Completion from JSON Validity (Model Output Quality)

- **Date:** 2026-08-10
- **Status:** Accepted
- **Trigger:** `reviews/phase_0_adversarial.md` Finding 5 (Codex adversarial review, Phase 0). ACCEPT. Reinforces ADR-004.

**Context:**
`EXPERIMENT_SPEC.md` §8a-3 (prior version) required the Base model to "return parseable output" as a pipeline-completion condition. However, a correctly functioning Base model breaking JSON on 1 out of 100 samples can be the result of inference, storage, invalid-detection, and zero-scoring all working correctly. In this case, pipeline completion would nonetheless be judged "not achieved," contradicting the principle ADR-004 intended (separating engineering-achievement conditions from scientific outcomes) — conversely, a broken pipeline that always returns `{}` would satisfy the "parseable" condition.

**Decision:**
Redefine the pipeline-completion condition as: "raw output was obtained for every sample," "the parser classified each sample as valid/invalid," and "both valid and invalid cases are recorded in the evaluation artifacts." The JSON validity rate and any scoring based on it are treated only as a metric under §8b (model performance improvement). Technical soundness checks are based on record counts, sample-ID alignment, non-missing raw output, exception logs, and artifact integrity.

**Alternatives Considered:**
- Keeping "parseable output" as a pipeline-completion condition (the prior plan): rejected, as it contradicts the intent of ADR-004 (treating pipeline completion independently of whether the numbers are good or bad).

**Consequences:**
- `EXPERIMENT_SPEC.md` §8a (items 3, 7) was updated with an added note.
- Does not overwrite ADR-004 (a reinforcing, concretizing relationship).

---

## ADR-012: Make LoRA Target-Module and Tower Selection an Explicit Approval Gate

- **Date:** 2026-08-10
- **Status:** Accepted
- **Trigger:** `reviews/phase_0_adversarial.md` Finding 8 (Codex adversarial review, Phase 0). ACCEPT. Reinforces ADR-003.

**Context:**
`EXPERIMENT_SPEC.md` §5 (prior version) listed LoRA target modules only as a set of suffix-name candidates like `q_proj`, without specifying whether the target is the language tower, the vision encoder, or the projector. Because Qwen3-VL has independent configs for the language and vision sides, specifying modules by suffix name alone risks unintentionally applying LoRA to identically-named layers on the vision side. This would change VRAM usage, the trainable parameter count, and the substantive meaning of the experiment.

**Decision:**
As a Phase 1 exit condition, finalize the fully-qualified names of the LoRA target modules, the target tower, the adapter parameter count, the freeze status of vision/projector, and whether assistant-only label masking is used, based on confirming the live model's structure, and require this to be recorded in an ADR as an approval gate rather than a mere name check.

**Alternatives Considered:**
- Finalizing based on suffix-name matching alone (the prior plan): rejected, since the risk of unintended application to the wrong module cannot be excluded.

**Consequences:**
- `EXPERIMENT_SPEC.md` §5, §10 / `IMPLEMENTATION_PLAN.md` Phase 1 exit conditions were updated to align with this ADR.
- Does not overwrite ADR-003 (a reinforcing, concretizing relationship).

---

## ADR-013: Strictly Pin the Base Artifact's Revision and Quantization Config Between Base and Fine-tuned

- **Date:** 2026-08-10
- **Status:** Accepted
- **Trigger:** `reviews/phase_0_adversarial.md` Finding 9 (Codex adversarial review, Phase 0). ACCEPT. Reinforces ADR-006.

**Context:**
`EXPERIMENT_SPEC.md` §2 stated that the choice between runtime quantization and a pre-quantized artifact (e.g., the Unsloth version) would be decided in Phase 1, but did not specify that, once decided, both Base evaluation and Fine-tuned training must use the artifact at exactly the same revision. Even if both are nominally "4-bit NF4," if the quantization parameters or processor revision differ, the measured difference would no longer be attributable solely to the effect of the LoRA adapter.

**Decision:**
After the base-artifact loading path is selected in Phase 1, pin the Hub revision of that model, processor/tokenizer/chat template, quantization config (NF4, double-quant, compute dtype), and attention backend to a commit SHA. Fine-tuned is this pinned artifact with the adapter applied, not a separately reloaded "equivalent" artifact. Add a check that compares config hashes at evaluation time to mechanically verify there is no difference other than adapter presence.

**Alternatives Considered:**
- Relying solely on the description "the same 4-bit NF4" (the prior plan): rejected, since it risks missing revision drift or processor differences.

**Consequences:**
- `EVALUATION_PROTOCOL.md` §2 / `EXPERIMENT_SPEC.md` §9, §10 were updated to align with this ADR.
- Does not overwrite ADR-006 (a reinforcing, concretizing relationship).

---

## ADR-014: Add a Production-Shape VRAM Go/No-Go Gate to Phase 1, Independent of the Smoke Test

- **Date:** 2026-08-10
- **Status:** Accepted
- **Trigger:** `reviews/phase_0_adversarial.md` Finding 10 (Codex adversarial review, Phase 0). ACCEPT.

**Context:**
The smoke test in `IMPLEMENTATION_PLAN.md` Phase 3 was designed to confirm only "it does not crash" using a configuration with reduced image resolution and rank, not just reduced record count and step count. A smoke-test success with short receipts, low resolution, and a small rank cannot detect an OOM under the production configuration (maximum image token count, the actual target token length, the actual rank, peak memory after optimizer-state allocation). Qwen3-VL-4B is distributed in BF16, and T4 (Turing) does not support the standard FlashAttention-2 kernels — an additional real constraint.

**Decision:**
As a Phase 1 exit condition, add a go/no-go gate that measures actual peak VRAM after forward/backward execution and optimizer-state allocation, using the production-intended image size, sequence length, LoRA rank, and microbatch size. Also measure the p50/p95/max image-token count and target-token count. Since T4 does not support FlashAttention-2, an explicit fallback path such as SDPA must be prepared and measured on that path. Phase 3's smoke test keeps the shape (image size, sequence length, rank, microbatch) confirmed by this go/no-go gate, and reduces only the record count and step count.

**Alternatives Considered:**
- Substituting the current smoke test (reduced shape) alone: rejected, since it cannot verify peak memory under the production shape.

**Consequences:**
- `IMPLEMENTATION_PLAN.md` Phase 1 and Phase 3 / `EXPERIMENT_SPEC.md` §10 were updated to align with this ADR.
- Whether the 4B model fits in 16GB (T4) is not asserted before this gate is actually measured.

---

## ADR-015: Pin the Hub Revisions of the Model, Dataset, and Processor to Commit SHAs

- **Date:** 2026-08-10
- **Status:** Accepted
- **Trigger:** `reviews/phase_0_adversarial.md` Finding 11 (Codex adversarial review, Phase 0). ACCEPT.

**Context:**
`EXPERIMENT_SPEC.md` §7 and `EVALUATION_PROTOCOL.md` §8 required recording the config hash, project commit hash, and package versions, but did not address pinning the Hub revisions of the model/dataset/processor or the dataset fingerprint. Left following the `main` branch, the same repo ID could later have different weights, chat template, preprocessing, or parquet files, turning a later reproduction run into a different experiment without anyone noticing.

**Decision:**
Pin the Hub revisions of the model, processor, dataset, and (if used) quantization artifact to specific commit SHAs rather than following the `main` branch. Record the dataset fingerprint, split counts, and content hashes in the result artifacts wherever feasible.

**Alternatives Considered:**
- Recording only the repo ID (the prior plan): rejected, since it cannot prevent silent changes in experimental conditions from later unannounced updates.

**Consequences:**
- `EXPERIMENT_SPEC.md` §7, §10 / `IMPLEMENTATION_PLAN.md` Phase 1 exit conditions were updated to align with this ADR.

---

## ADR-016: Adopt a Single-Seed Training Strategy and Scope the Conclusion Accordingly

- **Date:** 2026-08-10
- **Status:** Accepted
- **Trigger:** `reviews/phase_0_adversarial.md` Finding 12 (Codex adversarial review, Phase 0). USER DECISION REQUIRED → decided by the user.

**Context:**
Codex's finding: since the magnitude of improvement may depend strongly on the seed due to LoRA initialization and data ordering, a single-seed training result alone cannot support a claim of "improvement that generally reproduces." Because multi-seed training would consume several times the GPU budget, Claude Code did not decide this unilaterally and instead deferred the choice to the user ([[reviews/phase_0_adversarial.md]] §11, AGENTS.md §11 USER DECISION REQUIRED).

**Decision:**
The user prioritized conserving the GPU budget and chose to adopt single-seed training (approving the option presented as the recommended default). Accordingly, the final report's conclusion must explicitly state that it is "a conditional comparison for this fixed seed/checkpoint," and must not claim an improvement that generally reproduces across seeds ([[EXPERIMENT_SPEC]] §8b, §9).

**Alternatives Considered:**
- Multi-seed training: would allow assessing cross-seed variance and increase the generality of the conclusion, but the user did not adopt it due to the multi-fold GPU time cost.
- Deciding later: an option to defer until just before Phase 4 was also available, but the user decided immediately within this session.

**Consequences:**
- This item is removed from the open items in `docs/STATE.md`, and the Phase 4 training design assumes a single seed.
- If GPU budget allows in the future, retraining with additional seeds can be proposed as a new ADR (this ADR does not foreclose that option).

---

## ADR-017: Fix the Improvement Threshold X at 0.05

- **Date:** 2026-08-12
- **Status:** Accepted
- **Trigger:** `reviews/phase_2_adversarial.md` Finding 16 (Codex adversarial review, Phase 2 entry gate). USER DECISION REQUIRED → decided by the user.

**Context:**
`EXPERIMENT_SPEC.md` §8b defines "improvement achieved" as `CI_lower(Δ) ≥ X` on the held-out test set, but X had never been fixed — it was an open item from Phase 0 onward. ADR-009 requires it to be pre-registered before any Phase 2 performance output is observed. The adversarial review additionally found that the draft proposal justified candidate values partly by how likely each was to pass ("almost certain to pass"), which is the wrong basis for a pre-registered threshold: X must be chosen from what a given delta *means* for this task, not from a forecast of the result.

**Decision:**
`X = 0.05`. Improvement is declared only when the lower bound of the 95% paired-bootstrap confidence interval of Δ (the per-receipt difference in the primary metric, Fine-tuned − Base) reaches +0.05. This commits the conclusion to a difference large enough to be worth the fine-tuning cost, while remaining resolvable at `n = 100`. Per `EVALUATION_PROTOCOL.md` §6 this value may not be revisited after any result is seen.

**Alternatives Considered:**
- `X = 0.00` (improvement whenever the CI excludes zero): claims only a direction, makes no claim about effect size; not adopted.
- `X = 0.10`: a stronger claim, but at `n = 100` the CI width alone could prevent a real, moderate improvement from being declared; not adopted.

**Consequences:**
- `EXPERIMENT_SPEC.md` §8b and §10 and `EVALUATION_PROTOCOL.md` §6 must record X = 0.05 when the Phase 2 pre-registration is promoted.
- The threshold is expressed in units of the primary metric fixed by ADR-020.

---

## ADR-018: Run the ADR-014 VRAM Gate on the Free-Tier T4 First, Then Decide the Tier

- **Date:** 2026-08-12
- **Status:** Accepted
- **Trigger:** `reviews/phase_2_adversarial.md` Findings 20 and 21 (Codex adversarial review, Phase 2 entry gate). USER DECISION REQUIRED → decided by the user.

**Context:**
`EXPERIMENT_SPEC.md` §10 lists the Colab tier as an open item. It is not a scheduling detail: the tier determines the compute dtype (Tesla T4 is Turing/sm_75 and has **no bfloat16 support**, while the model is distributed in bfloat16), the attention backend (T4 also rules out the standard FlashAttention-2 kernels, per ADR-014), the image-token budget, and every number the VRAM go/no-go gate produces. The bf16 constraint was confirmed during the Phase 2 gate review and was not previously recorded anywhere in `docs/`.

**Decision:**
Run the ADR-014 production-shape VRAM go/no-go gate on the free-tier T4 first, and decide the tier from that measurement. If the gate returns NO-GO, the documented fallback ladder for image resolution is applied and the gate re-run; if it still returns NO-GO, an upgrade to an Ampere-or-newer tier (L4 / A100) is reconsidered as a new decision. Until and unless that happens, the experiment is pinned to the T4 path: `float16` compute **and** load dtype, and `attn_implementation="sdpa"`, applied identically to the Base and Fine-tuned conditions.

**Alternatives Considered:**
- Committing to the free T4 outright: rejected as premature — no production-shape memory measurement exists yet, so "it fits" would be an assumption.
- Moving to Colab Pro (L4/A100) now: would unlock bf16 and remove the fp16 loss-scaling concern, but the user chose to establish the measurement first rather than pay for headroom that may not be needed.

**Consequences:**
- The compute/load dtype is an experimental condition. Changing tiers later changes the numerical path and therefore requires a new ADR, not a silent configuration edit.
- The gate must measure the evaluation path (generation prefill and KV cache at maximum input plus maximum output length) as well as the training path, and report against the tier's actual free memory rather than a fixed constant.
- Because the fp16 path needs loss scaling, the gate must also confirm the loss is finite and the gradient scaler is not in persistent overflow.

---

## ADR-019: Freeze the Cross-Split Duplication Handling Policy Before the Audit Runs

- **Date:** 2026-08-12
- **Status:** Accepted
- **Trigger:** `reviews/phase_2_adversarial.md` Findings 3, 4 and 13 (Codex adversarial review, Phase 2 entry gate). USER DECISION REQUIRED → decided by the user. Concretizes ADR-008.

**Context:**
ADR-008 requires a cross-split duplication audit and requires its handling policy to be pre-registered *before* the audit is run, so the policy cannot be chosen once the duplicate count is known. The draft proposal violated this by defaulting to "report only," justified by the claim that a train↔test duplicate "inflates both conditions, not one." The adversarial review established that this claim is **false**: the Fine-tuned condition has trained on the duplicate and the Base condition has not, so a memorized duplicate raises the Fine-tuned score selectively and manufactures exactly the apparent improvement the held-out split exists to rule out. Pairing the predictions on the same test row does not remove that training exposure. The review also found that near-duplicate template clusters break the i.i.d. assumption underlying receipt-level bootstrap resampling.

**Decision:**
Frozen before the audit executes:
1. **Exact duplicates** (identical decoded RGB pixel content, hashed together with image dimensions and mode) found between `train` and `test` are **deterministically excluded from the test evaluation set**. The same applies to `validation`↔`test` exact duplicates.
2. **Near-duplicate / template clusters** are not excluded. Instead they define resampling groups, and the paired bootstrap becomes **group-aware**: clusters, not individual receipts, are the resampling unit, so correlated receipts cannot be counted as independent evidence.
3. The audit also covers `train`↔`validation`, to detect selection bias in checkpoint and hyperparameter choice.
4. The exclusion count, the cluster structure, and the resulting effective sample size are reported in `results/report.md` in every case, including when nothing is excluded.
5. The audit publishes **aggregate counts and a frozen automated verdict only** — never hashes, pair IDs, or candidate renderings — because a test-side hash can be joined against an already-viewed train or validation image and thereby reveal test content (ADR-008 test blindness).

**Alternatives Considered:**
- Report only, leaving the split untouched and downgrading the confirmatory claim if duplicates are found: keeps ADR-007's split inviolable, but knowingly leaves a leakage channel open in the primary comparison; not adopted.
- Group-aware reconstruction of all three splits: strongest against contamination, but departs furthest from CORD's official split and forfeits comparability with other work on this dataset; not adopted.

**Consequences:**
- This departs from ADR-007's fixed `test = 100` count whenever an exact duplicate is found. That departure is authorized by this ADR; the realized test count is recorded in the result artifacts, and the decision rule is applied to the reduced set.
- `EVALUATION_PROTOCOL.md` §6 must be updated to specify the group-aware paired bootstrap when the pre-registration is promoted.
- The audit must be implemented and executed as a Phase 1 exit condition, before Phase 2.

---

## ADR-020: Adopt Strict Verbatim Transcription as the Evaluation Estimand

- **Date:** 2026-08-12
- **Status:** Accepted
- **Trigger:** `reviews/phase_2_adversarial.md` Findings 8, 9 and 12 (Codex adversarial review, Phase 2 entry gate). USER DECISION REQUIRED → decided by the user.

**Context:**
The draft proposal instructed the model to "copy values verbatim ... do not convert, round, or reformat numbers," and simultaneously specified NFKC normalization and internal-whitespace collapse before scoring. The adversarial review found these contradictory — the instruction and the metric were measuring different tasks — and separately found that TED-Acc's alignment with the task had been asserted rather than established: Donut's TED uses character edit cost, so a materially wrong amount such as `58,000` versus `59,000` can cost a single character edit while a field-exact view treats the value as simply wrong.

**Decision:**
The estimand is **strict verbatim transcription**: the model must reproduce the value exactly as printed on the receipt, and the metric must agree with that definition.
- No NFKC normalization, no case folding, no internal-whitespace collapse. Only leading/trailing whitespace is stripped.
- Digit grouping, currency symbols, and full-width/half-width forms remain unnormalized, as the draft already proposed.
- **TED-Acc remains the primary metric**, so `EVALUATION_PROTOCOL.md` §5's stated leading candidate is retained and Δ stays well defined per receipt.
- A **per-receipt field-exact-match score is added as a mandatory secondary guardrail**, reported alongside TED-Acc for both conditions with its own paired confidence interval. Its purpose is to make TED-Acc's insensitivity to single-character amount errors visible rather than hidden.
- Before any model output exists, both candidate metrics are probed on synthetic error cases (wrong amount digit, missing field, extra field, item reorder, invalid JSON, near-correct OCR) and the results are recorded in `EVALUATION_PROTOCOL.md` §5.1. This probe requires no model and no test data.

**Alternatives Considered:**
- Keeping the verbatim estimand but switching the primary metric to a per-receipt field-exact score: would treat a one-digit amount error as unambiguously wrong, but departs from the metric `EVALUATION_PROTOCOL.md` has named as the leading candidate since Phase 0 and would require newly defining hierarchical and repeated-row handling from scratch; not adopted.
- Redefining the task as semantic extraction and enumerating every accepted equivalence: internally consistent, but any omission from the equivalence list silently becomes a scoring hole; not adopted.

**Consequences:**
- `EVALUATION_PROTOCOL.md` §5.1's normalization rules and `EXPERIMENT_SPEC.md` §4's prompt wording must be aligned to this estimand when the pre-registration is promoted.
- ADR-017's `X = 0.05` is expressed in TED-Acc units.
- The secondary field-exact metric is a reported guardrail, not part of the ADR-017 decision rule.
