# Phase 1 Closure and Phase 2 Pre-Registration — PROPOSAL (DRAFT v2)

> **THIS DOCUMENT IS NOT AUTHORITATIVE.**
> It is a *proposal* to close the remaining Phase 1 exit conditions in
> `docs/IMPLEMENTATION_PLAN.md` and to complete the ADR-009 pre-registration that must be finalized
> **before any performance output is observed in Phase 2**. Nothing here takes effect until it has
> passed the ADR-005 adversarial review and been promoted into `docs/EXPERIMENT_SPEC.md` /
> `docs/EVALUATION_PROTOCOL.md` / `docs/DECISIONS.md` (`AGENTS.md` §37).

- **Date:** 2026-08-12
- **Author:** Claude Code (orchestration / specification layer)
- **Phase:** Phase 1 closure → Phase 2 entry gate
- **Supersedes:** v1 (commit `f179dbd`), which the ADR-005 review returned **BLOCK** on

## Revision history

**v1 → v2.** v1 was reviewed adversarially on 2026-08-12 (`reviews/phase_2_adversarial.md`, Codex/Sol
role, read-only). Verdict **BLOCK**: 25 findings (17 `required`), all dispositioned **ACCEPT**. v2
resolves every one of them and incorporates the four user decisions now recorded as ADR-017 through
ADR-020. Findings are cited inline as `[F<n>]` so each resolution is traceable to what prompted it.

Two of v1's own claims were **wrong** and are called out here rather than quietly dropped:

- v1 argued a train↔test duplicate "inflates both conditions, not one." False — the Fine-tuned
  condition trained on it and the Base condition did not, so a memorized duplicate raises one side
  selectively `[F3]`. Superseded by ADR-019.
- v1 dismissed few-shot partly on optimizer-state memory, but evaluation-time inference retains no
  optimizer `[F23]`. The dismissal now rests only on claims that survive scrutiny, and the estimand
  is scoped accordingly (§5.1).

## User decisions incorporated

| ADR | Decision |
|---|---|
| ADR-017 | Improvement threshold **X = 0.05** |
| ADR-018 | Run the VRAM gate on the **free-tier T4 first**, then decide the tier; T4 path (fp16 + SDPA) pinned until it says otherwise |
| ADR-019 | **Exact cross-split duplicates excluded**; near-duplicate clusters become **group-aware bootstrap** units; policy frozen before the audit runs |
| ADR-020 | Estimand is **strict verbatim transcription**; TED-Acc stays primary, with a mandatory per-receipt **field-exact guardrail** |

---

## 0. Why this document exists

`docs/STATE.md` records Phase 1's environment/dataset sub-scope as complete with a genuine
`COLAB PASS` on real GPU hardware (Tesla T4, `cuda_available: True`, `Qwen3VLProcessor` loaded).
That closes the *execution* half of Phase 1.

It does not close Phase 1. `IMPLEMENTATION_PLAN.md` §2 lists five further exit conditions prefixed
"do not proceed to Phase 2 until all are met", and `EXPERIMENT_SPEC.md` §8b sets the pre-registration
deadline **before observing any performance output in Phase 2 — including baseline output on a
train/validation subset**. Phase 2 is therefore gated on decisions, not on more code.

**Evidence status.** Every factual claim in §1–§3 was read from the live Hugging Face Hub at the SHAs
in §1 (`config.json`, `model.safetensors.index.json`). Nothing is recalled from memory. Nothing was
executed on a GPU; items needing GPU measurement are marked and deferred to the §8.2 gate. Claims
that remain unverified are labelled **[HYPOTHESIS]** `[F25]`.

---

## 1. ADR-015 — Revision pinning

| Artifact | Repo ID | Commit SHA |
|---|---|---|
| Model + processor + tokenizer + chat template | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Dataset | `naver-clova-ix/cord-v2` | `7f0115a4b758a71d6473b8d085751692da2fef98` |

The model repo ships weights, processor, tokenizer and `chat_template.json` together, so one SHA pins
all of them — which is what ADR-013 requires.

**P-1:** pin both in `configs/*.yaml`, thread them through `load_cord_v2()` and the model/processor
loading path, and record both in every result artifact together with the dataset fingerprint and row
counts `[F17]`.

---

## 2. ADR-013 — Base artifact and quantization

**P-2: runtime-quantize the plain `Qwen/Qwen3-VL-4B-Instruct` with bitsandbytes NF4.** With runtime
quantization there is exactly one artifact and one SHA, so ADR-013's same-revision requirement is
satisfied structurally rather than by a check. A third-party pre-quantized re-upload would add a
second SHA and a second party able to change quantization parameters.

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: float16      # ADR-018; T4 has no bf16
model_load_dtype: float16              # [F20] non-quantized modules too, NOT the config's bfloat16
attn_implementation: sdpa              # ADR-018; T4 rules out FlashAttention-2
```

**P-2b `[F20]`:** `attn_implementation="sdpa"` selects the SDPA *interface*, not a specific kernel —
PyTorch dispatches among available backends, so a hardware change alters the numerical path while the
YAML still reads `sdpa`. The run must therefore **record** the realized SDPA backend, the GPU name and
capability, and a dump of every module's realized dtype, into the result artifact. Both conditions
must show an identical realized path.

---

## 3. ADR-012 — LoRA approval gate

### 3.1 Actual module structure (713 tensors, read from the pinned index)

```
model.language_model.embed_tokens                     (weight tied to the lm_head weight)
model.language_model.layers.{0..35}.self_attn.{q_proj,k_proj,v_proj,o_proj}
model.language_model.layers.{0..35}.self_attn.{q_norm,k_norm}
model.language_model.layers.{0..35}.mlp.{gate_proj,up_proj,down_proj}
model.language_model.layers.{0..35}.{input_layernorm,post_attention_layernorm}
model.language_model.norm

model.visual.patch_embed.proj
model.visual.pos_embed
model.visual.blocks.{0..23}.attn.{qkv,proj}           <- fused qkv
model.visual.blocks.{0..23}.mlp.{linear_fc1,linear_fc2}
model.visual.blocks.{0..23}.{norm1,norm2}
model.visual.merger.{linear_fc1,linear_fc2,norm}
model.visual.deepstack_merger_list.{0..2}.{linear_fc1,linear_fc2,norm}
```

**Narrow claim `[F19]`:** none of the **seven projection basenames** selected in §3.2 occurs anywhere
in the vision tower. (The towers *do* share other basenames — both contain modules named `norm` — so
the broader claim v1 made was wrong.) Qwen3-VL does have an `lm_head` **module**; it is its *weight*
that is tied to `embed_tokens`, which is why no `lm_head.weight` tensor appears in the checkpoint.

### 3.2 P-3 — Target tower and modules

**Language tower only.** Vision encoder, `merger` and `deepstack_merger_list` are frozen.

```yaml
target_modules: "model\\.language_model\\.layers\\.\\d+\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)"
```

PEFT accepts a regex string for `target_modules` and applies it as a full match against the module
key, so anchoring on `model.language_model.` makes a vision-side match impossible regardless of future
basename collisions. Excluded deliberately: `embed_tokens` (tied to the output head; `modules_to_save`
on a tied 151936×2560 embedding costs ~389 M fp16 parameters and defeats QLoRA on a T4), all norms,
and `q_norm`/`k_norm`.

**Rationale, and its status `[F25]`:** the working premise is that this task is decoder-side —
Qwen3-VL already reads receipt text, and what must be learned is faithful emission of *this* schema.
That premise is a **[HYPOTHESIS]**: no full-model inference has been run and no performance output
observed, so the vision tower's adequacy is unverified. Language-only LoRA may improve schema emission
while leaving OCR errors untouched. Vision/projector LoRA is therefore recorded as an **alternative
excluded on parameter and activation budget**, not as technically irrelevant, and the Phase 6
conclusion is scoped to decoder-side adaptation rather than to QLoRA adaptation in general.

### 3.3 P-4 — Rank, alpha, dropout, adapter size

Per-layer `sum(in + out)` over the seven projections is 57,344 (attention 20,480 + MLP 36,864) across
36 layers, so the adapter has `57,344 × 36 × r` parameters.

```yaml
lora_r: 16
lora_alpha: 32                 # scaling = alpha/r = 2
lora_dropout: 0.05
lora_bias: none
lora_use_rslora: false         # [F14] explicit, not left to library default
lora_use_dora: false           # [F14]
lora_init_lora_weights: true   # [F14] default Kaiming/zeros init, pinned explicitly
adapter_params: 33_030_144     # ~0.8% of the ~4B base; asserted at runtime in §8.2
```

Identical across smoke / mini / full, per ADR-014.

### 3.4 P-5 — Compute path (ADR-018)

The model ships bfloat16; **T4 is Turing (sm_75) and has no bf16 support.** Therefore fp16 compute
*and* load dtype (§2), fp16 mixed precision with the gradient scaler enabled, and SDPA. The §8.2 gate
must confirm the loss is finite and the scaler is not in persistent overflow. A later tier change
alters an experimental condition and requires a new ADR (ADR-018).

### 3.5 P-6 — Assistant-only label masking

Loss is computed only over the assistant turn (target JSON plus EOS); system prompt, user turn and all
image-token positions are set to `-100`. With ~1024 image tokens against a few hundred target tokens,
not masking would put most of the loss on reproducing a fixed prompt. Verified by unit test (§9).

---

## 4. Token budget — derived jointly, not fixed independently `[F2]`

v1 fixed `max_seq_len: 2048` here and `max_new_tokens: 768` in §5.3 **independently**, leaving only
256 tokens for the prompt, image boundary tokens and chat-template tokens. That would let training
silently drop target suffixes and EOS, and let evaluation cut a correct JSON object before its closing
brace and score it zero — a harness-manufactured *non*-improvement.

**P-7: the budget is derived from measurement, in this order, on train + validation only.**

```yaml
# Step 1 — pinned caps, expressed in the processor's ACTUAL parameter names, which take
# TOTAL PIXEL COUNTS (the shipped defaults are longest_edge: 16777216, shortest_edge: 65536).
processor_size:
  longest_edge:  1_048_576     # <= 1024 merged image tokens
  shortest_edge:   200_704     # >=  196 merged image tokens

# Step 2 — measured, not assumed [F18]: from the pinned processor's real image_grid_thw,
# dividing spatial patches by merge_size**2. NOT from pixel-area arithmetic.
measured_image_tokens:   {p50: TBD, p95: TBD, max: TBD}
measured_prompt_tokens:  {max: TBD}       # fully rendered chat template, not the bare string
measured_target_tokens:  {p50: TBD, p95: TBD, max: TBD}

# Step 3 — derived
max_new_tokens: ceil(measured_target_tokens.max * 1.25)
max_seq_len:    measured_image_tokens.max + measured_prompt_tokens.max + max_new_tokens + 64
```

**Hard assertions, all failing loudly `[F2]`:** zero training-target truncations; EOS present on every
training target; zero silent input truncations; the generation termination reason recorded per sample
(§9). Reaching `max_new_tokens` at test time is **model behaviour, not an execution defect**, and can
never justify a rerun with a larger cap `[F6]`.

If the §8.2 gate returns NO-GO, the fallback ladder is `longest_edge: 524_288` → `262_144`, applied
identically to both conditions, re-measured, and recorded before any performance is observed.

---

## 5. ADR-009 — Prompt, shots, and the full input freeze

### 5.1 P-8 — Zero-shot, both conditions, with the estimand scoped

**Zero-shot.** A demo requires a demo *image*, costing another ~1024 image tokens; two demos roughly
triple the visual context. Zero-shot also removes ADR-009's demo-ID and demo-selection pre-registration
requirements and a class of asymmetry risk.

`[F23]` v1 also cited optimizer-state memory, which is wrong — inference retains no optimizer. Whether
few-shot is feasible at all is a **[HYPOTHESIS]** about image-context and KV-cache memory, to be
measured in the evaluation arm of the §8.2 gate rather than asserted.

**Estimand, stated before any result exists:** this experiment measures **the adapter effect under the
frozen zero-shot prompt**, not a general Base-versus-Fine-tuned comparison. Zero-shot understates the
Base model's achievable performance and therefore biases the comparison *in favour of* the fine-tuned
model. Scoping the claim is the remedy; a caveat added afterwards would not be.

### 5.2 P-9 — Prompt-selection procedure

**One candidate, fixed a priori, no search.** This is the strictest option under ADR-009 and removes
the risk of a search that implicitly tunes the Base condition.

### 5.3 P-10 — The template (ADR-020: strict verbatim)

*System:*

```
You are an information extraction model. You read a receipt image and return the extracted
information as a single JSON object. You return only JSON. You never return explanations,
commentary, or markdown code fences.
```

*User (image attached as the message's image content):*

```
Extract the structured information from this receipt image as a single JSON object.

Use exactly these top-level keys when the corresponding information is present, and omit a key
entirely when it is not present:
- "menu": a list of ordered items. Each item may contain "nm" (name), "cnt" (count),
  "price", "unitprice", "discountprice", and "sub" (a list of sub-items with the same fields).
- "sub_total": may contain "subtotal_price", "discount_price", "service_price", "tax_price",
  "etc".
- "total": may contain "total_price", "cashprice", "changeprice", "creditcardprice",
  "menuqty_cnt", "menutype_cnt".
- "void_menu": voided items, same structure as "menu".

Transcribe every value exactly as printed on the receipt, character for character. Preserve digit
grouping such as "58,000" or "23.000", currency symbols, capitalization, and spacing exactly as
shown. Do not convert, round, reformat, translate, or tidy any value. Do not invent fields that
are not visible on the receipt.

Return only the JSON object.
```

The key inventory is provisional and must be verified mechanically against **train + validation only**
(never `test` — ADR-008); if any key outside it occurs, the template is corrected before the
pre-registration is frozen (§9 item 1). The wording is aligned to ADR-020's verbatim estimand.

### 5.4 P-10b — The complete input freeze `[F1]`

A byte-identical prompt string does **not** freeze the model input. All of the following are
pre-registered, and both conditions must resolve to identical values:

```yaml
message_construction:
  chat_template: from the pinned model repo's chat_template.json  # not a hand-written string
  apply_chat_template:
    tokenize: false                 # render to text, then process with the processor
    add_generation_prompt: true     # evaluation
    # training renders the same prefix, then appends the assistant turn + EOS
  content_order: [image, text]      # image content block first, then the instruction text
  target_serialization:             # canonical, so the training target is deterministic
    json_dumps:
      ensure_ascii: false
      separators: [",", ":"]        # compact; no incidental whitespace to learn
      sort_keys: false              # preserve CORD's key order
processor:
  padding: false                    # batch size 1 at eval; no padding side ambiguity
  truncation: false                 # never silently truncate; assert instead [F2]
generation_config:                  # complete, no library defaults left implicit
  do_sample: false
  num_beams: 1
  temperature: null
  top_p: null
  top_k: null
  repetition_penalty: 1.0
  max_new_tokens: <derived in §4>
  min_new_tokens: 0
  use_cache: true
  cache_implementation: static
  eos_token_id: from the pinned tokenizer
  pad_token_id: from the pinned tokenizer
  stop_strings: null                # no early stopping on text
output_decoding:
  slice: generated token ids only, excluding the prompt prefix
  skip_special_tokens: true
  clean_up_tokenization_spaces: false
```

**Assertion, run before evaluation begins `[F1]` `[F22]`:** for every sample, the Base and Fine-tuned
paths produce byte-identical input token IDs, attention masks and `image_grid_thw`, and structurally
identical resolved `GenerationConfig` and processor configuration. The **only** permitted difference is
the adapter state. v1's "no condition-dependent branch anywhere" was an impossible criterion `[F22]`;
the accurate rule is a one-item allowlist plus a structured diff of everything else. Implementation
preference: one quantized base instance with the PEFT adapter enabled/disabled, so the two conditions
cannot diverge by construction.

---

## 6. EVALUATION_PROTOCOL §5.1 — normative metric definitions

### 6.1 P-11 — Metrics (ADR-020)

- **Primary:** TED-Acc, per receipt.
- **Mandatory secondary guardrail:** per-receipt **field-exact-match** score, reported with its own
  paired CI. Its job is to expose TED-Acc's insensitivity to single-character amount errors
  (`58,000` vs `59,000` costs one character edit but is a materially wrong amount) `[F8]`.
- Also reported: raw JSON validity, recoverable-payload parse rate, strict schema validity, micro-F1,
  Exact Match.

**P-11b — synthetic-error probe, before any model output exists `[F8]`.** Both candidate metrics are
scored on constructed cases — wrong amount digit, missing field, extra field, item reorder, invalid
JSON, near-correct OCR — and the resulting table is transcribed into `EVALUATION_PROTOCOL.md` §5.1.
This needs no model and no test data, and it converts "TED-Acc is appropriate" from an assertion into
evidence.

### 6.2 P-12 — Donut evaluator, vendored and pinned `[F11]`

Vendor `JSONParseEvaluator` from `clovaai/donut` (`donut/util.py`) into
`src/vlm_lab/third_party/donut_eval.py` at a pinned commit SHA, with its license.

v1 claimed that "vendoring at a pin means there are no differences to document." That is **false**:
§6.3 wraps the evaluator in our own extraction, normalization and invalid-output rules, making the
whole thing a custom pipeline whose differences must be documented. Requirements:

1. Record the exact Donut commit SHA in `EVALUATION_PROTOCOL.md` §5.1 and in every result artifact.
2. Transcribe the pinned source's **actual** behaviour into §5.1 in prose — field flattening, key
   accumulation, `menu` ordering, duplicate-row handling, the F1 aggregation level, and the TED
   normalization denominator. Not asserted from memory here; it is a §9 deliverable.
3. Pin **both** `zss` **and `nltk`** with exact `==` versions — official `donut/util.py` imports
   `nltk.edit_distance` directly, which v1 missed.
4. Freeze regression fixtures with known scores, so a dependency drift cannot silently change the
   metric.
5. Call it from a single shared function in `src/vlm_lab/evaluation.py` used by both conditions.

### 6.3 P-13 — Rules around the metric (ADR-020)

| Rule | Specification |
|---|---|
| Raw JSON validity `[F10]` | Whether the **unmodified** output parses via `json.loads`. Reported separately. Fenced output is **not** raw-valid — the prompt forbids fences. |
| Recoverable payload `[F10]` | If and only if the entire output is wrapped in one markdown fence, strip it and re-parse. Reported as a separate rate. Content metrics are computed from the **recovered** payload; both rates are published so a Base/Fine-tuned difference in fencing habits is visible rather than absorbed. |
| Parse failure | Scores 0 on TED-Acc, field-exact, F1 and Exact Match. Never dropped from the denominator (ADR-011). |
| Non-object top level | Parse failure. |
| Strict schema validity `[F7]` | Checked against the §6.5 schema, reported separately from raw validity. A parseable but schema-invalid object still receives content credit; that is stated explicitly rather than left implicit. |
| String normalization (ADR-020) | **Strip leading/trailing whitespace only.** No NFKC, no case folding, no internal-whitespace collapse, no digit-grouping or currency normalization. v1's NFKC + whitespace collapse contradicted the prompt's verbatim instruction `[F9]`. |
| Missing vs. null | Absent key and key-present-with-`null` are treated as identical on both sides. |
| Empty prediction `{}` | Raw-valid JSON; scores near 0; counted valid. |
| Micro-F1 `[F12]` | Global micro-F1 over flattened field–value pairs (Donut's definition). It has no additive per-sample score, so it is **incompatible with the `mean(Δ_i)` construction** — not impossible to bootstrap. If a CI is wanted, use a paired receipt-level **cluster** bootstrap that recomputes the corpus statistic per replicate. |
| Bootstrap statistic | TED-Acc and field-exact are per-receipt, so `Δ_i` is well defined for both. |

### 6.4 P-14 — Paired bootstrap (ADR-019)

```yaml
bootstrap_B: 10000
bootstrap_method: percentile
bootstrap_ci: 0.95
bootstrap_seed: 20260812
bootstrap_unit: duplication_cluster   # ADR-019 group-aware; singleton receipts are their own cluster
```

**Group-aware resampling `[F13]`:** i.i.d. receipt-level resampling would be invalid if the test set
contains near-duplicate template clusters — correlated receipts counted as independent evidence narrow
the CI and inflate the false "improvement achieved" rate. Per ADR-019 the resampling unit is the
duplication cluster from the frozen audit. If the audit finds no multi-member clusters, every cluster
is a singleton and this degenerates exactly to the ordinary paired receipt bootstrap; that evidence is
recorded either way.

**Percentile over BCa `[F24]`:** chosen for transparency and implementation simplicity — one fewer
thing that can be silently wrong in the statistic the conclusion hinges on. v1's claim that BCa's
correction is "second-order at n=100" was unsupported and is withdrawn; sample size alone does not
establish negligible bias/skew correction for a bounded, possibly zero-inflated Δ distribution. The
trade-off is recorded honestly as a simplicity choice.

### 6.5 P-14b — Normative output schema `[F7]`

A key inventory is not a schema. Frozen from **train + validation only**, before Phase 2:

- The recursive structure: allowed top-level keys; which are objects, which are arrays of objects;
  the permitted keys at each level including nested `sub`.
- Value types: all leaves are **strings** in CORD's ground truth; a numeric leaf in a prediction is a
  schema violation (though the value may still match after Donut's stringification — hence the
  separate strict-validity report).
- Unknown keys: permitted structurally but counted as false positives by the field metrics, and
  recorded as a schema violation.
- Scalar-vs-list shape for `menu` / `void_menu` / `sub`: always lists after `convert_ground_truth`;
  a bare object in a prediction is a schema violation.

Expressed as an explicit JSON-Schema document under `configs/` so it is machine-checkable, not prose.

### 6.6 P-15 — Seed

```yaml
seed: 42            # LoRA init, data ordering, any sampling; single seed per ADR-016
```

---

## 7. Decision rule, test budget, and the training configuration

### 7.1 Decision rule (ADR-017)

`X = 0.05`, on TED-Acc, against `CI_lower(Δ) ≥ X` from the §6.4 group-aware paired bootstrap.
Regression is `CI_upper(Δ) < 0`. Field-exact is reported alongside as a guardrail but is **not** part
of the decision rule (ADR-020).

### 7.2 P-16 — Test-execution budget `[F6]`

**One** execution, in Phase 5, covering both conditions in the same run.

- **Infrastructure/artifact failure** (runtime disconnect, crash, corrupted or incomplete artifact) —
  the only condition permitting a rerun, and only as an **exact-config replay**.
- **Generation-length termination** (outputs reaching `max_new_tokens`) is model behaviour, **not** a
  defect, and never grounds for a rerun or a larger cap.
- **Code/config defect discovered mid-run**: the run is voided, the defect and its evidence are
  recorded in `docs/STATE.md` *before* any rerun, and the rerun uses the corrected config recorded in
  advance.
- Outputs stay **sealed** until both conditions and the artifact-integrity checks have completed, so a
  Base-complete / Fine-tuned-failed run cannot expose Base test results during debugging.
- Every attempted test-split access is logged.

### 7.3 P-17 — Search space and split roles `[F14]` `[F15]`

v1's "everything except the two knobs below is fixed" was false. The complete training configuration:

```yaml
# --- searched (the ONLY free parameters) ---
search_space:
  learning_rate: [1e-4, 2e-4]
  num_train_epochs: [2, 3]
max_trials: 4                        # full 2x2 grid; no adaptive search, no extra rounds

# --- fixed [F14] ---
optimizer: paged_adamw_8bit          # bitsandbytes; chosen for T4 memory headroom (ADR-018)
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1e-8
weight_decay: 0.0
max_grad_norm: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
per_device_train_batch_size: 1
gradient_accumulation_steps: 8       # effective batch 8
gradient_checkpointing: true
per_device_eval_batch_size: 1
packing: false
group_by_length: false
dataloader_shuffle_seed: 42
save_strategy: epoch
eval_strategy: epoch
fp16: true
bf16: false
```

**Split roles, stated unambiguously `[F15]`:** every trial **fits only on `train`**; `validation` is
**evaluation-only and never updates weights**; `test` is untouched until Phase 5. The selected
configuration is **not** retrained afterwards — the selected checkpoint is used directly, so no
post-selection refit can leak validation information.

**Pre-existing documentation defect to fix `[F15]`:** `IMPLEMENTATION_PLAN.md` contradicts itself —
Phase 4's description says `qwen_cord_mini` "references only the validation split", while the configs
table says "train subset + validation". The authoritative reading is the one above, and the plan must
be corrected when the pre-registration is promoted.

### 7.4 P-18 — Checkpoint selection

Evaluate on `validation` at the end of every epoch; select the highest validation TED-Acc; tie-break
toward **fewer training steps**; **no early stopping** (every trial runs its full planned epochs), so
the rule stays mechanical and no patience hyperparameter needs pre-registering.

### 7.5 P-18b — Environment pinning `[F14]`

`pyproject.toml` currently pins `datasets`, `transformers`, `torch`, `pillow`, `torchvision` — and
contains **neither `peft` nor `bitsandbytes`**, both of which this design depends on. Required before
Phase 2:

- Add exact `==` pins for `peft`, `bitsandbytes`, `accelerate` (if used by the trainer), `pyyaml`,
  `zss` and `nltk`.
- Direct pins do not freeze transitive resolution across Colab sessions, so also produce a **resolved
  lock artifact** (e.g. `uv pip compile` / `pip freeze` output) committed alongside, and record its
  hash in every result artifact.

---

## 8. Remaining executable Phase 1 work

### 8.1 P-19 — ADR-008 / ADR-019 duplication audit

Implemented in `src/vlm_lab/` with a thin notebook caller. Policy is **frozen by ADR-019 before the
audit runs**; the audit executes it, it does not choose it `[F3]`.

**Signals:**

- **Exact image duplication:** SHA-256 over decoded RGB pixel bytes **plus image dimensions and mode**
  `[F4]` (not file bytes, which vary with re-encoding).
- **Near duplicates:** dHash (64-bit) with Hamming distance ≤ 3, forming clusters.
- **Ground-truth duplication, two distinct signals `[F4]`:** (a) a **value-inclusive** canonical JSON
  hash, which detects genuinely duplicated annotations; and (b) the type-only structure signature,
  which detects shared *templates* and is reported separately as template-cluster evidence. v1
  conflated these.

**Coverage `[F4]`:** `train`↔`test`, `validation`↔`test` (validation drives prompt and checkpoint
selection, so a validation↔test duplicate is also contamination), and `train`↔`validation` (selection
bias).

**Actions (ADR-019):** exact cross-split duplicates are excluded from the test evaluation set;
near-duplicate clusters become the §6.4 bootstrap resampling units; exclusion counts, cluster
structure and the resulting effective sample size are always reported.

**Test blindness `[F4]`:** the audit publishes **aggregate counts and a frozen automated verdict
only** — never hashes, pair IDs, or candidate renderings, because a test-side hash can be joined
against an already-viewed train/validation image and thereby reveal test content.

### 8.2 P-20 — ADR-014 / ADR-018 VRAM gate `[F21]`

`notebooks/01b_vram_gate.ipynb`, Phase 1, on the free-tier T4 (ADR-018). v1's single 13.0 GiB
criterion measured only a training step; `max_memory_reserved()` from training says nothing about
generation prefill/KV-cache peaks or non-PyTorch allocations, and the 1.56 GiB residual was an
unsupported margin. Replaced by **four separately-measured arms**:

| Arm | What it measures |
|---|---|
| A — Load | model load + NF4 quantization; realized dtypes per module; realized SDPA backend `[F20]` |
| B — Train | forward + backward + `optimizer.step()` across a **full accumulation window** (8 microbatches) with the exact §7.3 optimizer, at production shape, gradient checkpointing on |
| C — Checkpoint | adapter save **and reload**, since ADR-018's T4 path makes resume-from-checkpoint load-bearing |
| D — Eval | Base **and** Fine-tuned generation at maximum input + maximum output length, including KV cache |

Each arm reports `max_memory_allocated`, `max_memory_reserved`, and **device free/total** after
warm-up. **GO criterion:** every arm's peak reserved memory fits within the **measured free memory of
the actual device**, with the residual reported rather than assumed. Scale reference: fp32 Adam
moments for 33,030,144 parameters are ~252 MiB before gradients, master weights, activations and
allocator effects — which is why the optimizer is now pinned (§7.3) rather than left unspecified.

The gate also produces:

- the ADR-012 approval-gate evidence by **execution**: the real trainable-parameter count asserted
  equal to 33,030,144, and every adapted module name asserted to start with `model.language_model.`;
- the §4 token distributions from the processor's real `image_grid_thw` over the full 900-row
  train+validation corpus `[F18]`;
- confirmation that the fp16 loss is finite and the grad scaler is not in persistent overflow;
- a feasibility datapoint for few-shot evaluation memory `[F23]`.

NO-GO triggers the §4 fallback ladder, a re-run, and re-recording — all before any performance output.

### 8.3 P-21 — Structurally enforced test blindness `[F5]`

`load_cord_v2()` returns a `DatasetDict` containing all three splits, so "Phase 2 does not read test"
currently depends on caller discipline, and a caller can materialize test before deciding not to index
it. Required change: a **split-scoped loader** (explicit allowed-splits argument) that **fails closed**
on `"test"`, with Phase 2 requesting only `train`/`validation`, plus a unit test asserting no
test-split request is issued.

### 8.4 P-22 — Per-sample evidence artifact `[F17]`

Aggregate `metrics.json` + `report.md` cannot support an audit of paired statistics: a wrong join, a
dropped parse failure, or a duplicated sample still yields plausible aggregates. Pre-register an
immutable per-sample record containing: stable sample ID, condition, raw output, generation
termination reason, raw-validity and recoverable-parse status, parsed object, reference linkage hash,
every per-sample metric, exception state, duplication-cluster ID, and config/artifact hashes.

**Assert one-to-one Base / Fine-tuned / reference alignment before bootstrapping.**

---

## 9. Deliverables before pre-registration can be declared complete

Items 1–4 are mechanical and touch train/validation only (ADR-008-safe).

1. **Schema inventory and normative JSON Schema** (§5.3, §6.5) from train + validation.
2. **Token distributions** (§4) from the processor's real `image_grid_thw`; derive `max_new_tokens`
   and `max_seq_len`.
3. **Donut evaluator transcription** (§6.2 req. 2) plus the frozen regression fixtures.
4. **Synthetic-error metric probe** (§6.1 P-11b) and its results table.
5. **Duplication audit implemented and run** (§8.1), executing ADR-019's frozen policy.
6. **Split-scoped loader** (§8.3) with its test.
7. **Environment pins and lock artifact** (§7.5).
8. **VRAM gate executed on T4** (§8.2), yielding GO/NO-GO plus the ADR-012 approval evidence.
9. **Promotion:** accepted content merged into `EXPERIMENT_SPEC.md` §4/§5/§8b/§10,
   `EVALUATION_PROTOCOL.md` §5.1/§6, `IMPLEMENTATION_PLAN.md` (including the §7.3 contradiction fix),
   and new ADRs; `configs/qwen_cord_{smoke,mini,full}.yaml` created; `docs/STATE.md` updated.

Only after item 9 may `02_baseline.ipynb` be executed — `EXPERIMENT_SPEC.md` §8b forbids observing any
performance output before the pre-registration exists.

**Acceptance criteria for the eventual implementation task:**

- The §5.4 input-equality assertion passes: identical token IDs, masks, `image_grid_thw` and resolved
  configs across conditions, with adapter state the single allowlisted difference `[F22]`.
- Label masking covered by a unit test asserting `-100` at all non-assistant positions including image
  tokens.
- Every parameter in this document read from `configs/*.yaml`; no magic numbers in notebooks
  (`AGENTS.md` §24).
- Nothing in Phase 2 can request the test split (§8.3).

---

## 10. Summary

| # | Item | Class |
|---|---|---|
| P-1 | Pin model/dataset SHAs | proposal |
| P-2 / P-2b | Runtime NF4; fp16 load+compute; SDPA; record realized path | proposal (ADR-018) |
| P-3 | LoRA on the language tower only, regex-anchored | proposal (ADR-012 gate) |
| P-4 | r=16, α=32, dropout 0.05, explicit init flags; 33,030,144 params | proposal |
| P-5 / P-6 | fp16 path; assistant-only label masking | proposal |
| P-7 | Token budget derived jointly from measurement | proposal |
| P-8 / P-9 / P-10 | Zero-shot, single a-priori prompt, verbatim wording | proposal (ADR-020) |
| P-10b | Complete input freeze + equality assertion | proposal |
| P-11 / P-11b | TED-Acc primary + field-exact guardrail; synthetic-error probe | proposal (ADR-020) |
| P-12 | Donut vendored at a pin; `zss` **and** `nltk`; fixtures | proposal |
| P-13 | Parsing / verbatim normalization / validity-split rules | proposal (ADR-020) |
| P-14 | B=10000, percentile, 95%, **group-aware** units | proposal (ADR-019) |
| P-14b | Normative JSON Schema | proposal |
| P-15 | Seed 42 | proposal (ADR-016) |
| P-16 | One test execution; rerun conditions narrowed | proposal |
| P-17 | Full training config; split roles; plan contradiction fix | proposal |
| P-18 / P-18b | Checkpoint selection; environment pins + lock | proposal |
| P-19 | Duplication audit executing ADR-019 | proposal (ADR-019) |
| P-20 | Four-arm VRAM gate against measured free memory | proposal (ADR-014/018) |
| P-21 | Split-scoped, fail-closed loader | proposal |
| P-22 | Per-sample evidence artifact + alignment assertion | proposal |

**All four previously-open user decisions are now closed** by ADR-017 through ADR-020.

**Gate status:** ADR-005 adversarial review of **v2** — PENDING. v1's review returned BLOCK
(`reviews/phase_2_adversarial.md`); this revision resolves all 25 findings. v2 must not be promoted,
and `02_baseline.ipynb` must not be executed, until the re-review passes.
