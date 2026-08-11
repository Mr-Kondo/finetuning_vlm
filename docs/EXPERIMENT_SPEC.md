# EXPERIMENT_SPEC.md — Experiment Specification

**Status:** Draft — Codex adversarial review completed (Phase 0)
**Related:** [[DECISIONS]] ADR-001–016 / [[EVALUATION_PROTOCOL]] / [[IMPLEMENTATION_PLAN]] / [[STATE]]

## 1. Purpose and Scope

### Purpose (Hypothesis)
Fine-tuning `Qwen/Qwen3-VL-4B-Instruct` on the CORD v2 train split with 4-bit QLoRA
significantly improves structured receipt information extraction (image → JSON) accuracy
on the held-out test split, compared to the base model before fine-tuning (zero/few-shot).

### In Scope
- Validation on a single model (Qwen3-VL-4B-Instruct) and a single dataset (CORD v2).
- Validating the effect of a single training method: 4-bit QLoRA.
- Building an experiment pipeline that runs entirely on Google Colab.

### Non-goals
- Comparison against other models or sizes (2B/8B/32B, etc.).
- Comparison against full fine-tuning or other PEFT methods (unquantized LoRA, Prefix Tuning, etc.).
- Validating generalization to datasets other than CORD.
- Production deployment or serving performance (throughput optimization, etc.).
- Exhaustive hyperparameter search (limited to the three stages smoke/mini/full).

## 2. Target Model

- **HF repo ID:** `Qwen/Qwen3-VL-4B-Instruct`
- **Confirmed facts (2026-08-10, HF Hub):** task=image-text-to-text, library=transformers, license=apache-2.0, downloads=3.6M. Existence confirmed.
- **Candidate loading paths:** whether to load the plain `Qwen/Qwen3-VL-4B-Instruct` with runtime 4-bit quantization via `transformers` + `bitsandbytes`, or to use the pre-quantized `unsloth/Qwen3-VL-4B-Instruct-bnb-4bit` (existence confirmed), will be compared in Phase 1 (decided based on measured VRAM and load speed, with an ADR added).
- **Definition of the base model:** the above model without fine-tuning (regardless of quantization, it must be evaluated under the same quantization settings as the fine-tuned side — [[EVALUATION_PROTOCOL]] §4).

## 3. Dataset

- **HF repo ID:** `naver-clova-ix/cord-v2`
- **Confirmed structure (2026-08-10, HF Hub, config name: `"default"`, verified directly against the HF Dataset Viewer API's `/splits` and `/size` endpoints on 2026-08-11 — the dataset has no `configs:` block in its README frontmatter and no per-config file layout, so `datasets.load_dataset("naver-clova-ix/cord-v2", ...)` resolves the single config automatically with no `name=` argument needed; this corrects an earlier, unverified assumption in `reviews/phase_0_adversarial.md` §7 that the config name was literally `cord-v2`):**

  | split | rows | columns |
  |---|---:|---|
  | train | 800 | image, ground_truth |
  | validation | 100 | image, ground_truth |
  | test | 100 | image, ground_truth |

- **`ground_truth` format:** a Donut-format JSON string (a nested structure containing `gt_parse`. E.g., hierarchical fields such as `menu.nm` / `menu.cnt` / `menu.price` / `sub_total` / `total`).
- **Required conversion (Phase 1, responsibility of `data.py`):** convert Donut's s-tag/nested format into a "clean JSON schema" for instruction tuning. Because this conversion logic itself is used as the shared ground truth for both Base/Fine-tuned evaluation, its correctness determines the reliability of the entire evaluation.
- **Fixed split usage:**
  - `train` (800 rows): used only for QLoRA training.
  - `validation` (100 rows): used only for hyperparameter selection, prompt tuning, and checkpoint selection ([[DECISIONS]] ADR-007).
  - `test` (100 rows): used only for the final Phase 5 comparison (Base vs Fine-tuned). Never referenced at any training or tuning stage.
- **Test blindness principle ([[DECISIONS]] ADR-008):** the `test` split's images and `ground_truth` must not be subject to human visual/manual inspection at any development stage, except for the mechanical bulk processing in Phase 5. Development, debugging, and visual inspection of the `data.py` conversion logic must be done using `train`/`validation` only; only automated processing is applied to `test` once the conversion logic is finalized (in Phase 1, only mechanical checks such as record count and parse success/failure are allowed).
- **Cross-split duplication audit ([[DECISIONS]] ADR-008):** split-name separation alone cannot prevent leakage if images or receipt templates are duplicated (near-identical) between `train` and `test`. In Phase 1, an automated audit using image hashes/perceptual hashes and normalized hashes of the ground-truth structure will be performed, and the duplicate threshold and handling policy (exclusion / group-aware re-split / reporting only) will be pre-registered if duplicates are found. The actual number of duplicates is unconfirmed at audit time.

## 4. Task Definition / Input-Output Format

- **Input:** one receipt image + a fixed instruction prompt (e.g., "extract structured information from the image as JSON").
- **Output:** a JSON string conforming to the converted schema from §3.
- **Prompt template:** finalized in Phase 1 and recorded in `configs/*.yaml`. Once decided, the exact same template is used in all three of training, Base evaluation, and Fine-tuned evaluation ([[DECISIONS]] ADR-006). No wording differences are permitted between the system prompt and user prompt either.

## 5. Training Method: 4-bit QLoRA

- **Quantization:** NF4, 4-bit (bitsandbytes).
- **Adapter:** LoRA (PEFT). Base weights are frozen.
- **Target modules:** candidate attention/MLP projection layers in Qwen3-VL (e.g., `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`). Since the actual module names depend on the model implementation, they will be confirmed against the live model's module list in Phase 1, finalized, and recorded in an ADR.
- **Rank / alpha / dropout:** config-driven hyperparameters, with separate values for the smoke/mini/full config stages (the values themselves are decided in Phase 1–3).
- **Other:** gradient checkpointing and gradient accumulation are assumed for VRAM savings.

## 6. Execution Environment Constraints (Google Colab)

- **GPU tier (typical offerings as of 2026-08; the actual allocation must be checked each time):** free tier T4 (16GB); A100 (40GB) or L4 (22GB), etc. on Colab Pro/Pro+.
- **Assumption:** single GPU. Since session time limits and forced disconnects can occur, the design assumes resumption from checkpoints.
- **Disk:** confirm that model weights, quantization caches, and checkpoints fit within Colab's disk/Drive quota.
- **Unconfirmed:** the Colab tier the user will actually use and its actual VRAM amount (§10).

## 7. Reproducibility Requirements

- All hyperparameters must be described in `configs/*.yaml`; hardcoding into notebooks/code is prohibited.
- Fix the random seed, and record config, git commit hash, and seed in the result artifacts.
- Pin dependency package versions (`pyproject.toml`, created in Phase 1).
- Decoding at evaluation time must be deterministic (greedy or temperature=0, `max_new_tokens` fixed), using exactly the same decoding settings between Base and Fine-tuned.
- The Hub revisions of the model, processor/tokenizer, quantization adapter (if used), and dataset must be pinned to specific commit SHAs rather than following the `main` branch ([[DECISIONS]] ADR-015). The pinned revisions and the dataset fingerprint (row counts, content hashes) must be recorded in the result artifacts.

## 8. Success Conditions — Defined as Two Independent Conditions ([[DECISIONS]] ADR-004)

### 8a. Pipeline Completion (Engineering Achievement Condition, Blocking)

"Pipeline complete" is declared only once **all** of the following are satisfied. This is unrelated to whether the numeric results are good or bad.

1. Environment setup (dependency installation, model/dataset download) completes without error.
2. Loading and format conversion of CORD v2 by `data.py` completes without error across all splits. Schema validation of the converted JSON (structure/types are as expected) is confirmed on `train`/`validation`; for `test`, the finalized conversion logic is applied mechanically and only record count and parse success/failure are checked automatically (no visual inspection — [[DECISIONS]] ADR-008).
3. Baseline inference with the Base model runs without crashing on a small subset of `train`/`validation` (Phase 2, not including test), and raw output is obtained for every sample.
4. The QLoRA smoke run (a tiny subset, only a few steps) completes without OOM/crash, and the adapter can be saved.
5. The full QLoRA training run (mini → full) completes through the planned number of steps/epochs, and checkpoints can be saved.
6. The saved fine-tuned adapter can be loaded and inference can be run.
7. The evaluation script processes the entire held-out test set for both Base and Fine-tuned, obtains raw output for every sample, the parser classifies each as valid/invalid, and both results (regardless of whether the values themselves are good or bad) are recorded in the metrics artifact (metrics.json + report.md) under `results/`.

If even one item is not met, the pipeline is considered incomplete, and the model performance improvement evaluation cannot be carried out (since the evaluation itself would be meaningless).

**Note ([[DECISIONS]] ADR-011):** whether the output parses as valid JSON is itself a model performance metric and is not included among the pass/fail conditions for pipeline completion. As long as samples that fail to parse are still recorded and classified without crashing, the pipeline for that run is considered complete. The JSON validity rate is treated as a metric under §8b (model performance improvement).

### 8b. Model Performance Improvement (Scientific Success Condition, Outcome-Dependent)

- **Pre-registration timing ([[DECISIONS]] ADR-009):** all of the following elements must be finalized and appended to this document and to [[EVALUATION_PROTOCOL]] **before observing any performance output in Phase 2 (including baseline output on the train/validation subset)**. The standard is "before observing performance even once," not "before training is executed." Using validation to choose the **values** of hyperparameters, etc. is permitted (ADR-007), but the **selection procedure** itself must be fixed before any values are chosen.
- **Elements requiring pre-registration (currently undecided — §10):**
  1. The complete formula for the primary metric, X (the improvement threshold), and the decision rule (§8b below)
  2. Whether the Base model is zero-shot or few-shot. If few-shot, the source of the demos (train only) and the specific list of demo IDs
  3. The prompt-selection procedure (deciding a single candidate manually, comparing multiple candidates on validation, etc.)
  4. The parser, normalization, and aggregation specification ([[EVALUATION_PROTOCOL]] §5, ADR-009)
  5. The hyperparameter search space and the upper bound on the number of trials
  6. The checkpoint-selection, early-stopping, and tie-breaking rules
  7. The number of test executions allowed and the conditions for re-running (in principle, once)
- **Decision rule:** on the held-out test set (100 samples), "improvement achieved" is declared when the lower bound of the confidence interval computed via paired bootstrap (resampling with pairing the receipt-level difference Δ = Fine-tuned − Base) satisfies `CI_lower(Δ) ≥ X` ([[EVALUATION_PROTOCOL]] §6, [[DECISIONS]] ADR-010). The decision is not made by whether the independent confidence intervals of Base and Fine-tuned overlap (since this is paired data over the same 100 samples).
- **Handling a "no improvement" result:** if `CI_lower(Δ) < X`, this is treated as "no improvement," a legitimate experimental result, and a root-cause analysis (insufficient data volume, insufficient epochs, task-formulation issues, etc.) is reported. This is not a project failure.
- **Handling regression:** if the Fine-tuned model significantly underperforms the Base model (e.g., `CI_upper(Δ) < 0`), this is reported separately as "regression" (which may suggest training collapse, overfitting, or a task-formulation error).
- **Scope of the conclusion ([[DECISIONS]] ADR-009, ADR-016):** training is run with a single fixed random seed (the user has decided to adopt a single-seed strategy — [[DECISIONS]] ADR-016). Accordingly, the final report's conclusion must explicitly state that it is "a conditional comparison for this fixed seed/checkpoint," and must not claim an improvement that generally reproduces across seeds.

## 9. Threats to Validity

- **Data leakage:** confusion between train/validation/test (structurally prevented by [[DECISIONS]] ADR-007).
- **Evaluation asymmetry:** the risk of overestimating the effect of fine-tuning by using a prompt/decoding configuration that disadvantages the Base model (prevented by [[DECISIONS]] ADR-006).
- **Small test-set variance:** since the test split has only 100 samples, the variance of point estimates is large. Addressed with bootstrap confidence intervals ([[EVALUATION_PROTOCOL]] §6).
- **Quantization noise:** since 4-bit quantization itself can affect output quality, both Base and Fine-tuned must be evaluated under identical quantization settings.
- **Configuration distortion due to VRAM constraints:** reducing batch size/sequence length/image resolution too far to fit Colab's constraints risks making training effectively non-functional. Detected early via the smoke test.
- **Prompt/chat-template mismatch:** if the chat template differs subtly between training time and evaluation time, the performance comparison is distorted.
- **Version drift between quantization/base artifacts:** even under "the same 4-bit NF4," runtime quantization and a pre-quantized artifact (e.g., the Unsloth version) can differ in quantization parameters or processor revision. It must be guaranteed that the base artifact used for Fine-tuned training and the base artifact used for Base evaluation are exactly the same revision ([[DECISIONS]] ADR-013).
- **Possible presence of CORD in pretraining data:** whether `naver-clova-ix/cord-v2` (or CORD-derived data) is included in Qwen3-VL's pretraining corpus is unconfirmed. Even if it were, since Base and Fine-tuned share the same base model, the validity of the paired difference Δ attributable to adding the adapter is not directly undermined — but there is a risk of a ceiling effect on the interpretation of "generalization to unseen receipts." This point is explicitly noted as a caveat in the final report.
- **Limitation of a single training seed:** depending on LoRA initialization and data ordering, the magnitude of improvement may depend strongly on the seed. When training with a single seed, the bootstrap confidence interval is conditional on that one training run and does not guarantee general reproducibility (see §8b).

## 10. Open Verification Items

To be finalized before or during Phase 1; add an ADR to [[DECISIONS]] as each item is finalized.

- [ ] The exact LoRA target module names, the target tower (language side / vision side / projector), the adapter parameter count, the freeze status of the vision/projector, and whether assistant-only label masking is used (confirmed against the live Qwen3-VL implementation and recorded as an approval gate — [[DECISIONS]] ADR-012).
- [ ] Initial values for LoRA rank / alpha / dropout (for each of the smoke/mini/full configs).
- [ ] The finalized wording of the prompt template (system/user).
- [ ] Image resolution and image token budget (since image tokens contribute significantly to context length for VLMs, this directly drives VRAM estimation).
- [ ] The primary metric, the improvement threshold X, and the number of paired-bootstrap trials/seed (§8b, [[EVALUATION_PROTOCOL]] §6).
- [ ] The Colab tier the user will actually use, and the production-shape (production-intended image size, sequence length, rank, microbatch) VRAM go/no-go gate measurement ([[DECISIONS]] ADR-014).
- [ ] Whether to use `unsloth/Qwen3-VL-4B-Instruct-bnb-4bit` or runtime-quantize the plain model, and, once selected, the revision-pinning and consistency-check method between Base and Fine-tuned ([[DECISIONS]] ADR-013).
- [ ] The pinned Hub revision (commit SHA) values for the model/processor/dataset ([[DECISIONS]] ADR-015).
- [ ] The results of the image/receipt-template duplication audit between `train`/`test`, and the handling policy if duplicates are found ([[DECISIONS]] ADR-008).
- [x] **Resolved by user decision:** whether to train with multiple seeds (to assess cross-seed variance) or to adopt a single seed with a scope-limited conclusion. The user selected the single-seed strategy ([[DECISIONS]] ADR-016). This did not block Phase 1 entry (it only affects the Phase 4 training design).
