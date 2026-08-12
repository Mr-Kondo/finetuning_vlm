# Phase 1 Closure and Phase 2 Pre-Registration — PROPOSAL (DRAFT)

> **THIS DOCUMENT IS NOT AUTHORITATIVE.**
> It is a *proposal* drafted by Claude Code to close the remaining Phase 1 exit conditions
> in `docs/IMPLEMENTATION_PLAN.md` and to complete the ADR-009 pre-registration that must be
> finalized **before any performance output is observed in Phase 2**.
>
> Nothing here takes effect until it has passed the ADR-005 adversarial review, the items marked
> `USER DECISION REQUIRED` have been decided by the user, and the accepted content has been promoted
> into `docs/EXPERIMENT_SPEC.md` / `docs/EVALUATION_PROTOCOL.md` / `docs/DECISIONS.md`
> (`AGENTS.md` §37: review evidence and proposals are not source of truth).

- **Date:** 2026-08-12
- **Author:** Claude Code (orchestration / specification layer)
- **Phase:** Phase 1 closure → Phase 2 entry gate
- **Blocks:** `IMPLEMENTATION_PLAN.md` Phase 2 ("do not proceed to Phase 2 until all are met")

---

## 0. Why this document exists

`docs/STATE.md` records this session's Phase 1 sub-scope as complete with a genuine `COLAB PASS`
on real GPU hardware (Tesla T4, `cuda_available: True`, processor `Qwen3VLProcessor` loaded).
That closes the *execution* half of Phase 1.

It does **not** close Phase 1. `IMPLEMENTATION_PLAN.md` §2 lists five additional Phase 1 exit
conditions, and prefixes them with "do not proceed to Phase 2 until all are met". Additionally,
`EXPERIMENT_SPEC.md` §8b sets the pre-registration deadline at
**"before observing any performance output in Phase 2 (including baseline output on the
train/validation subset)"** — i.e. before `02_baseline.ipynb` prints a single metric.

So Phase 2 is gated on design decisions, not on more code. This document proposes each of them.

**Evidence status:** every factual claim in §1–§3 was read from the live Hugging Face Hub at the
SHAs recorded in §1 (model `config.json`, `model.safetensors.index.json`). Nothing here is recalled
from memory or inferred from a summary. Nothing here was executed on a GPU — items requiring GPU
measurement are explicitly marked as such and deferred to the ADR-014 gate run.

---

## 1. ADR-015 — Revision pinning (commit SHAs)

**Status quo:** `load_cord_v2(revision=...)` accepts a revision but defaults to `None` (Hub default).
No SHA has been chosen. This is an open Phase 1 exit condition.

**Evidence (read from the Hub API, 2026-08-12):**

| Artifact | Repo ID | Commit SHA | Hub `lastModified` |
|---|---|---|---|
| Model + processor + chat template | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | 2025-10-15 |
| Dataset | `naver-clova-ix/cord-v2` | `7f0115a4b758a71d6473b8d085751692da2fef98` | 2022-07-19 |
| (Alternative, see §2) Pre-quantized | `unsloth/Qwen3-VL-4B-Instruct-bnb-4bit` | `79cd853d0bc98bb0d67f865123eb49ef8985c2ec` | 2025-10-14 |

Note the model repo ships the processor, tokenizer and `chat_template.json` in the *same* repo, so a
single SHA pins model weights, processor, tokenizer and chat template together — which is exactly
what ADR-013 requires.

**Proposal (P-1):** pin `model_revision: ebb281ec70b05090aa6165b016eac8ec08e71b17` and
`dataset_revision: 7f0115a4b758a71d6473b8d085751692da2fef98` in `configs/*.yaml`, thread them through
`load_cord_v2()` and the model/processor loading path, and record both in every result artifact.

**Disposition class:** Claude proposal — routine, factual, no experimental trade-off.

---

## 2. ADR-013 — Base artifact: runtime quantization vs. pre-quantized

**Status quo:** open. `EXPERIMENT_SPEC.md` §10 asks whether to use
`unsloth/Qwen3-VL-4B-Instruct-bnb-4bit` or to runtime-quantize the plain model.

**Proposal (P-2): runtime-quantize the plain `Qwen/Qwen3-VL-4B-Instruct` with bitsandbytes NF4.**

Rationale:

- ADR-013 requires that the base artifact used for Base evaluation and the one used for Fine-tuned
  training be *exactly the same revision*. With runtime quantization there is literally one artifact
  and one SHA, so that requirement is satisfied structurally rather than by a check.
- The pre-quantized repo is a third-party re-upload on its own release cadence; it adds a second SHA
  to keep in sync and a second party who can change quantization parameters.
- Unsloth's speed advantage matters most for full fine-tuning throughput; this experiment's dominant
  cost is a 4B model on 800 samples, and the project has no Unsloth dependency pinned today
  (`pyproject.toml` pins `transformers`/`torch`/`datasets`/`pillow`/`torchvision` only).

**Quantization config to pin (both conditions, identical):**

```yaml
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: float16   # NOT bfloat16 — see §3.4
```

**Disposition class:** Claude proposal. Reversible before Phase 3; if reversed, ADR-013's
same-revision requirement must be re-verified for the substitute repo.

---

## 3. ADR-012 — LoRA approval gate (fully-qualified modules, tower, param count, masking)

This is the ADR-012 *approval gate*. The evidence below was enumerated from
`model.safetensors.index.json` at the pinned SHA — i.e. the live model's real parameter names, not
suffix guesses (which is precisely what ADR-012 was created to prevent).

### 3.1 Actual module structure (713 tensors)

```
model.language_model.embed_tokens                     (tied with output head; no separate lm_head)
model.language_model.layers.{0..35}.self_attn.{q_proj,k_proj,v_proj,o_proj}
model.language_model.layers.{0..35}.self_attn.{q_norm,k_norm}
model.language_model.layers.{0..35}.mlp.{gate_proj,up_proj,down_proj}
model.language_model.layers.{0..35}.{input_layernorm,post_attention_layernorm}
model.language_model.norm

model.visual.patch_embed.proj
model.visual.pos_embed
model.visual.blocks.{0..23}.attn.{qkv,proj}           <- fused qkv, NOT q_proj/k_proj/v_proj
model.visual.blocks.{0..23}.mlp.{linear_fc1,linear_fc2}
model.visual.blocks.{0..23}.{norm1,norm2}
model.visual.merger.{linear_fc1,linear_fc2,norm}
model.visual.deepstack_merger_list.{0..2}.{linear_fc1,linear_fc2,norm}
```

**Material observation:** the vision tower uses a *fused* `attn.qkv` and `mlp.linear_fc1/fc2`, and
shares **no** module basename with the language tower. The specific failure ADR-012 was written to
prevent (a suffix like `q_proj` silently matching vision-side layers) therefore cannot occur for
`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` on this model. We still specify
fully-qualified targeting, because the gate requires it and because relying on that coincidence is
exactly the kind of hidden assumption the gate exists to catch.

### 3.2 Proposal (P-3): target tower and modules

**Target tower: the language tower only.** Vision encoder, `merger`, and `deepstack_merger_list`
are frozen.

Rationale: the task is not "learn to see receipts" — Qwen3-VL already reads receipt text (the base
model produces text from these images). The task is "emit *this* JSON schema faithfully", which is a
decoder-side behaviour. Freezing the vision tower also keeps the trainable count and the activation
memory down on a 16 GiB T4, and avoids perturbing the visual features that the Base condition also
depends on.

**Target modules (regex, anchored to the language tower):**

```
target_modules: "model\\.language_model\\.layers\\.\\d+\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)"
```

PEFT accepts a regex string for `target_modules`; this anchors on `model.language_model.` so no
vision-side module can match, regardless of future basename collisions.

Excluded deliberately: `embed_tokens` (tied to the output head — adapting it would change the output
projection too, and `modules_to_save` on a tied 151936×2560 embedding costs ~389M fp16 params, which
defeats the point of QLoRA on a T4), all norms, and `q_norm`/`k_norm`.

### 3.3 Proposal (P-4): rank / alpha / dropout and adapter parameter count

Per-layer `sum(in_features + out_features)` over the seven targeted projections is **57,344**
(attention 20,480 + MLP 36,864), across 36 layers. Adapter parameter count is therefore
`57,344 × 36 × r = 2,064,384 × r`:

| r | Adapter params (all 7 projections) | Adapter params (attention-only) |
|---|---|---|
| 8 | 16.52 M | 5.90 M |
| **16** | **33.03 M** | 11.80 M |
| 32 | 66.06 M | 23.59 M |

**Proposed values, identical across smoke / mini / full (ADR-014 requires the smoke config to keep
the production shape and reduce only record count and step count):**

```yaml
lora_r: 16
lora_alpha: 32          # alpha = 2r, the common default; scaling = alpha/r = 2
lora_dropout: 0.05
lora_bias: none
adapter_params: 33_030_144   # ~0.8% of the ~4B base
```

### 3.4 Proposal (P-5): compute dtype — a real T4 constraint

The model is distributed in **bfloat16** (`config.json`: `"dtype": "bfloat16"`). **Tesla T4 is Turing
(sm_75) and has no bfloat16 tensor-core support.** ADR-014 already flags that T4 rules out
FlashAttention-2; the bf16 point is the same class of constraint and is not yet recorded anywhere in
`docs/`.

Consequences to pin:

- `bnb_4bit_compute_dtype: float16`, and fp16 mixed precision (`fp16: true`, `bf16: false`).
- fp16 training needs loss scaling; the trainer's default grad scaler must be enabled, and the
  ADR-014 gate run must confirm the loss is finite and not producing persistent scaler overflow.
- Attention backend: **SDPA**, explicitly (`attn_implementation="sdpa"`), for *both* Base and
  Fine-tuned — ADR-006 fairness requires the identical backend on both sides.

If the user later chooses an Ampere-or-newer tier (L4 / A100), bf16 becomes available and this pin
should be revisited via a new ADR rather than changed silently — it is an experimental condition.

### 3.5 Proposal (P-6): assistant-only label masking

**Enable assistant-only label masking**: loss is computed only over the assistant turn (the target
JSON plus its EOS), with the system prompt, the user turn, and all image tokens set to `-100`.

Rationale: with ~1024 image tokens versus a few hundred target tokens (§4), *not* masking would put
the large majority of the loss on reproducing a fixed prompt and on image-token positions, drowning
out the signal that actually matters. This must be implemented as a verified property, not an
assumption — see the acceptance criteria in §9.

---

## 4. Image resolution and token budget

**Evidence:** Qwen3-VL uses `patch_size: 16` with `spatial_merge_size: 2`, so one visual token
covers a 32×32 px block. Real CORD images observed in the committed `01_dataset.ipynb` Colab run:

| Sample | Size | Pixels | Uncapped image tokens |
|---|---|---|---|
| `train[0]` | 864×1296 | 1,119,744 | 1,093 |
| `validation[1]` | 1108×1478 | 1,637,624 | 1,599 |

The processor's shipped default is `longest_edge: 16,777,216` pixels — i.e. effectively uncapped, up
to ~16k image tokens. Leaving that default is not viable on a T4 and would also make the Base and
Fine-tuned context lengths depend on incoming image size in an unbounded way.

**Proposal (P-7):**

```yaml
image_max_pixels: 1_048_576    # <= 1024 image tokens
image_min_pixels:   200_704    # >=  196 image tokens (guards against over-shrinking small receipts)
max_seq_len: 2048              # image tokens + prompt + target
```

Both conditions use identical values (ADR-006). `1,048,576` keeps typical CORD receipts at close to
native resolution (a 864×1296 receipt is barely downscaled) while bounding the visual context at
1024 tokens.

**This is a proposal contingent on measurement.** ADR-014's gate run must report the p50/p95/max
image-token count and target-token count under these settings, and the peak VRAM. If the gate fails,
the documented fallback ladder is `image_max_pixels: 524_288` (512 tokens) → `262_144` (256 tokens),
applied identically to both conditions, with the change recorded before any performance is observed.

**Disposition class:** Claude proposal, gated on the ADR-014 measurement.

---

## 5. ADR-009 — Prompt template and shot design

### 5.1 Proposal (P-8): zero-shot, both conditions

**Zero-shot for both Base and Fine-tuned.** No in-context demos.

Rationale: a few-shot demo for this task requires a *demo image*, which costs another ~1024 image
tokens per demo. Two demos would roughly triple the visual context and the VRAM, on the same T4 that
must also hold the 4-bit model, activations and optimizer state. The trade-off is real and it is
adverse: few-shot would most plausibly force `image_max_pixels` down for both conditions, degrading
OCR fidelity for both. Zero-shot also removes the ADR-009 requirement to pre-register a demo-ID list
and a demo-selection procedure, and removes a class of Base/Fine-tuned asymmetry risk.

Acknowledged cost: zero-shot understates the Base model's achievable performance, which biases the
comparison *in favour of* the fine-tuned model. This is a known limitation and must be stated in the
Phase 6 report as a scope limit on the conclusion ("Base is evaluated zero-shot; a few-shot Base
would likely score higher"). It is recorded here, before any result is seen, rather than discovered
afterwards.

### 5.2 Proposal (P-9): prompt-selection procedure

**One candidate, fixed a priori, no comparison.** The template below is adopted as written; no
validation-set prompt search is run.

This is the strictest available option under ADR-009 (which requires the *procedure* to be fixed
before any values are chosen). A validation-based prompt search would be permissible under ADR-007,
but it would mean observing performance output, which under ADR-009 may not happen until
pre-registration is complete — so it would have to be pre-registered here anyway, with a trial cap.
Fixing a single prompt is simpler, cheaper, and removes the risk of a search that implicitly tunes
the Base condition.

### 5.3 Proposal (P-10): the template

Used byte-identically in training, Base evaluation, and Fine-tuned evaluation (ADR-006 /
`EXPERIMENT_SPEC.md` §4).

*System:*

```
You are an information extraction model. You read a receipt image and return the extracted
information as a single JSON object. You return only JSON. You never return explanations,
commentary, or markdown code fences.
```

*User (with the image attached as the message's image content):*

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

Copy values verbatim as they appear on the receipt, including digit grouping such as "58,000"
or "23.000". Do not convert, round, or reformat numbers. Do not invent fields that are not
visible on the receipt.

Return only the JSON object.
```

**Open sub-item:** the key inventory above is transcribed from CORD v2's Donut schema as observed in
the two committed `01_dataset.ipynb` samples plus the CORD schema. Before adoption it must be
verified mechanically against the **train and validation splits only** (never `test` — ADR-008) that
no key outside this inventory occurs. If additional keys exist, the template is corrected *before*
pre-registration is frozen. This is a deliverable in §9, not an assumption.

**Decoding (identical both conditions, `EXPERIMENT_SPEC.md` §7):**

```yaml
do_sample: false          # greedy
temperature: null
top_p: null
max_new_tokens: 768
```

`max_new_tokens: 768` is provisional and must be set from the measured target-token distribution in
§9 (proposed rule: `ceil(p100_target_tokens × 1.25)` on train+validation), so that truncation cannot
silently penalize either condition.

---

## 6. EVALUATION_PROTOCOL §5.1 — normative metric definitions

This is the largest remaining pre-registration item and the one most able to invalidate the
experiment if left loose.

### 6.1 Proposal (P-11): primary metric

**Primary metric: TED-Acc.** Secondary/reported: field-level F1, JSON validity rate, Exact Match.
This adopts the "leading candidate" already named in `EVALUATION_PROTOCOL.md` §5.

### 6.2 Proposal (P-12): use the official Donut implementation, pinned

Use `JSONParseEvaluator` from the official Donut repository
(`https://github.com/clovaai/donut`, `donut/util.py`), **vendored into
`src/vlm_lab/third_party/donut_eval.py` at a pinned commit SHA**, rather than a re-implementation.
`EVALUATION_PROTOCOL.md` §5.1 explicitly permits this and requires that any custom implementation
document its differences — vendoring at a pin means there are no differences to document.

Requirements attached to this proposal:

1. The exact commit SHA of `donut/util.py` is recorded in `EVALUATION_PROTOCOL.md` §5.1 and in every
   result artifact.
2. Its actual behaviour — field flattening, key accumulation, ordering/sorting of `menu` items,
   duplicate-row handling, the F1 aggregation level, and the TED normalization denominator — is
   **read from the pinned source and transcribed into `EVALUATION_PROTOCOL.md` §5.1 in prose**, so
   that the protocol is self-contained and a future reader does not have to re-derive it. This
   transcription is a §9 deliverable; this proposal deliberately does **not** assert those details
   from memory.
3. Its transitive dependency (`zss`, for tree edit distance) is added to `pyproject.toml` with an
   exact `==` pin, consistent with `EXPERIMENT_SPEC.md` §7.
4. It is called from `src/vlm_lab/evaluation.py` through a single shared function used by both
   conditions (ADR-006), with no condition-dependent branch anywhere in the path.

### 6.3 Proposal (P-13): rules around the metric

These are *not* covered by Donut's evaluator and must be pinned by us:

| Rule | Proposal |
|---|---|
| JSON extraction from raw output | Strip a leading/trailing markdown fence (` ```json ` / ` ``` `) if and only if the whole output is wrapped in one; then `json.loads` the result. No brace-scanning, no repair, no retry. Identical for both conditions. |
| Parse failure | Sample scores 0 for TED-Acc, F1 and Exact Match, and is counted in the JSON validity rate (`EVALUATION_PROTOCOL.md` §7, ADR-011). Never dropped from the denominator. |
| Non-object top level (e.g. a JSON list or string) | Treated as a parse failure. |
| String normalization before comparison | Unicode NFKC, then strip leading/trailing whitespace, then collapse internal whitespace runs to a single space. **No** case folding, **no** digit-grouping normalization, **no** currency-symbol stripping — CORD ground truth stores values verbatim (`"58,000"`, `"23.000"`), so normalizing them would discard real signal and would reward a model that reformats. |
| Missing vs. null | A key that is absent and a key present with value `null` are treated as identical (both = absent) on both sides. |
| Empty prediction (`{}`) | Valid JSON; scores whatever the evaluator gives (near 0), counted as valid in the JSON validity rate. |
| F1 aggregation level | Global micro-F1 over flattened field–value pairs accumulated across all samples (Donut's own definition, per `EVALUATION_PROTOCOL.md` §5.1). Reported as such, explicitly, in `report.md`. |
| Per-sample score for the paired bootstrap | TED-Acc is per-sample by construction, so `Δ_i` is well defined. Field-level F1 is **not** per-sample under micro aggregation and therefore **cannot** be the bootstrap statistic; it is reported as a point estimate only. This is why TED-Acc, not F1, is the primary metric. |

### 6.4 Proposal (P-14): paired bootstrap configuration

```yaml
bootstrap_B: 10000
bootstrap_method: percentile     # not BCa
bootstrap_ci: 0.95
bootstrap_seed: 20260812         # distinct from the training seed
```

Percentile over BCa: percentile is what `EVALUATION_PROTOCOL.md` §6 lists as the default candidate,
is implementable in a few lines with no extra dependency, and is transparent. BCa's acceleration term
adds implementation surface (and a jackknife pass) for a correction that is second-order at
`n = 100` — not worth the extra thing that can be silently wrong in an experiment whose conclusion
hinges on it.

### 6.5 Proposal (P-15): training seed

```yaml
seed: 42
```

Single seed per ADR-016. Used for LoRA init, data ordering and any sampling; recorded in artifacts.

---

## 7. Decision rule, threshold X, and test-execution budget

### 7.1 `USER DECISION REQUIRED` — the improvement threshold X

`EXPERIMENT_SPEC.md` §8b declares improvement when `CI_lower(Δ) ≥ X` on the 100-sample held-out test
set, where `Δ = TED-Acc(Fine-tuned) − TED-Acc(Base)` per receipt. **X has never been chosen.**

Per `AGENTS.md` §17, an acceptance-threshold choice is explicitly a `USER DECISION REQUIRED` item;
Claude Code must not set it unilaterally. Options:

| Option | Meaning | Comment |
|---|---|---|
| `X = 0.00` | Any statistically significant improvement counts | Weakest claim; almost certain to pass if fine-tuning works at all, so it tests very little |
| **`X = 0.05` (recommended)** | The 95% CI lower bound must clear +5 TED-Acc points | Substantive but attainable; a real QLoRA run on CORD typically moves schema-adherence a long way from a zero-shot base |
| `X = 0.10` | CI lower bound must clear +10 points | Strong claim; risks declaring "no improvement" for a genuinely working pipeline given `n = 100` |

Whichever is chosen is frozen here and **may not be revisited after any result is seen**
(`EVALUATION_PROTOCOL.md` §6).

### 7.2 Proposal (P-16): test-execution budget

**One** execution of the held-out test evaluation, in Phase 5, covering both conditions in the same
run. A re-run is permitted **only** for a demonstrable execution defect (crash, truncated run,
corrupted artifact) — never because the numbers were unsatisfying — and any re-run must be recorded
in `docs/STATE.md` with the defect evidence before it happens.

### 7.3 Proposal (P-17): hyperparameter search space and trial cap

Everything except the two knobs below is fixed by this document. Search runs on `qwen_cord_mini`
against the **validation** split only (ADR-007), never test.

```yaml
search_space:
  learning_rate: [1e-4, 2e-4]
  num_train_epochs: [2, 3]
max_trials: 4                  # the full 2x2 grid; no adaptive search, no extra rounds
selection_metric: TED-Acc on validation
```

### 7.4 Proposal (P-18): checkpoint selection, early stopping, tie-breaking

- Evaluate on the validation split at the end of every epoch.
- Select the checkpoint with the highest validation TED-Acc.
- **Tie-break:** fewer training steps wins (prefer the earlier checkpoint).
- **Early stopping:** none. Every trial runs its full planned epoch count; this keeps the selection
  rule mechanical and removes a patience hyperparameter that would itself need pre-registering.

---

## 8. ADR-008 duplication audit and ADR-014 VRAM gate — remaining executable work

These two exit conditions cannot be closed by a document; they need code and, for ADR-014, a GPU.

### 8.1 Proposal (P-19): ADR-008 cross-split duplication audit

Implement in `src/vlm_lab/data.py` + a thin notebook/script caller:

- **Exact-image duplication:** SHA-256 over the decoded RGB pixel bytes (not the file bytes, which
  vary with re-encoding), computed for all 1000 rows; report any hash occurring in both `train` and
  `test`.
- **Near-duplicate images:** dHash (64-bit difference hash) with a Hamming-distance threshold of
  `<= 3`, reported as *candidates* — near-duplicate detection is heuristic and the audit reports, it
  does not silently delete.
- **Ground-truth structure duplication:** SHA-256 over the canonical JSON of the converted ground
  truth with all leaf *values* replaced by their type, catching receipts that share a template.
- **Test-blindness compliance:** the audit outputs **counts and hashes only** — no test images, no
  test ground-truth content, no per-sample test rendering (ADR-008).

**Handling policy proposal:** *report only* for this experiment; do not re-split and do not exclude
rows. Re-splitting would break the `train=800 / validation=100 / test=100` structure that ADR-007
fixes and that both conditions share; and since Base and Fine-tuned are compared *pairwise on the
same test rows*, a duplicate inflates both conditions, not one. If the audit finds exact
train↔test image duplicates, the count is reported in `report.md` as a stated caveat on
generalization, and the finding is escalated as a new `USER DECISION REQUIRED` item rather than
handled silently.

### 8.2 Proposal (P-20): ADR-014 production-shape VRAM go/no-go gate

Add `notebooks/01b_vram_gate.ipynb` (Phase 1, not Phase 3 — ADR-014 requires it independent of the
smoke test). It must, on a real Colab GPU:

1. Load the pinned model at the pinned SHA with the §2 quantization config and `attn_implementation="sdpa"`.
2. Apply the §3 LoRA config; **print the real trainable-parameter count and assert it equals
   33,030,144**, and assert every adapted module name starts with `model.language_model.`
   (this is the ADR-012 approval-gate evidence, produced by execution rather than by assertion in a doc).
3. Report the p50 / p95 / max image-token count and target-token count over a train/validation
   sample under §4's settings.
4. Run forward + backward + `optimizer.step()` at the **production shape** (`image_max_pixels`,
   `max_seq_len`, `lora_r`, microbatch size), with gradient checkpointing on, and report
   `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`.
5. Confirm the fp16 loss is finite and the grad scaler is not in persistent overflow (§3.4).
6. Emit an explicit **GO** / **NO-GO** verdict against a headroom criterion.

**Proposed criterion:** `max_memory_reserved() <= 13.0 GiB` on the 14.56 GiB T4 (~10% headroom for
allocator fragmentation and the eval-time KV cache). NO-GO triggers the §4 fallback ladder, re-run,
and re-record — before any performance output is observed.

**Proposed microbatch shape:** `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 8`
(effective batch 8), `gradient_checkpointing: true`. Per ADR-014 the Phase 3 smoke config must reuse
this exact shape and reduce only record count and step count.

### 8.3 `USER DECISION REQUIRED` — Colab tier for the training phases

Phase 1's own validation ran on a **free-tier T4 (14.56 GiB)**. Whether Phase 3–5 run there is a
separate, unmade decision with real consequences:

- **T4 (free).** Everything in this document is sized for it (fp16 not bf16, SDPA not FA2, 1024
  image tokens, microbatch 1). Risk: a 4B VLM over 800 samples × 2–3 epochs × up to 4 search trials
  is many hours on a T4, against free-tier session limits and forced disconnects.
  `EXPERIMENT_SPEC.md` §6 already assumes checkpoint resumption — on this tier that assumption
  becomes load-bearing, and resume-from-checkpoint must actually be exercised, not just configured.
- **L4 / A100 (Pro / Pro+).** Unlocks bf16 (and FA2 on A100), removes the fp16 loss-scaling concern,
  and makes the schedule comfortable — but changes the pinned compute dtype in §3.4, which is an
  experimental condition and needs a new ADR rather than a silent switch.

This affects §3.4, §4 and §8.2, so it should be decided **before** the ADR-014 gate is run — running
the gate on a tier the experiment will not use wastes the run.

---

## 9. Deliverables required before pre-registration can be declared complete

Ordered. Items 1–3 are mechanical and touch train/validation only (ADR-008-safe).

1. **Schema key inventory** (§5.3 open sub-item): enumerate every key occurring in the converted
   ground truth across train + validation; correct the prompt template if any key is outside the
   listed inventory.
2. **Token-length distribution** (§4, §5.3): p50/p95/max image tokens and target tokens under the
   §4 settings, on train + validation; set `max_new_tokens` from the measured rule.
3. **Donut evaluator transcription** (§6.2 requirement 2): read the pinned `donut/util.py` and
   transcribe its real behaviour into `EVALUATION_PROTOCOL.md` §5.1.
4. **Duplication audit implemented and run** (§8.1).
5. **ADR-014 gate notebook implemented and executed on the chosen Colab tier** (§8.2), yielding
   GO/NO-GO plus the ADR-012 approval-gate evidence.
6. **Promotion:** accepted content merged into `EXPERIMENT_SPEC.md` §4/§5/§8b/§10,
   `EVALUATION_PROTOCOL.md` §5.1/§6, `IMPLEMENTATION_PLAN.md`, and new ADRs in `DECISIONS.md`;
   `configs/qwen_cord_{smoke,mini,full}.yaml` created; `docs/STATE.md` updated.

Only after item 6 may `02_baseline.ipynb` be executed, because step 6 is what makes the
pre-registration real and `EXPERIMENT_SPEC.md` §8b forbids observing any performance output before
it exists.

**Acceptance criteria for the eventual implementation (for the Luna task, not for this document):**

- No condition-dependent branch anywhere in the inference/evaluation path (ADR-006); a test asserts
  Base and Fine-tuned take the identical code path with identical decoding settings.
- Label masking (§3.6) is covered by a unit test asserting that all non-assistant positions,
  including image-token positions, carry label `-100`.
- Every parameter in this document is read from `configs/*.yaml`; no magic numbers in notebooks
  (`AGENTS.md` §24).
- The test split is not read by anything in Phase 2.

---

## 10. Summary of dispositions requested

| # | Item | Class |
|---|---|---|
| P-1 | Pin model/dataset SHAs | Claude proposal |
| P-2 | Runtime NF4 quantization of the plain model | Claude proposal |
| P-3 | LoRA on the language tower only, regex-anchored | Claude proposal (ADR-012 gate) |
| P-4 | `r=16, alpha=32, dropout=0.05`, 33.03 M adapter params | Claude proposal |
| P-5 | fp16 compute dtype + SDPA (T4 has no bf16 / no FA2) | Claude proposal |
| P-6 | Assistant-only label masking | Claude proposal |
| P-7 | `image_max_pixels=1_048_576`, `max_seq_len=2048` | Claude proposal, gated on ADR-014 |
| P-8 | Zero-shot, both conditions | Claude proposal |
| P-9 | Single a-priori prompt, no search | Claude proposal |
| P-10 | The prompt template + greedy decoding | Claude proposal |
| P-11 | TED-Acc as primary metric | Claude proposal |
| P-12 | Vendor Donut's evaluator at a pinned commit | Claude proposal |
| P-13 | Parsing / normalization / aggregation rules | Claude proposal |
| P-14 | `B=10000`, percentile, 95%, seed 20260812 | Claude proposal |
| P-15 | Training seed 42 | Claude proposal |
| — | **Improvement threshold X** | **USER DECISION REQUIRED** |
| P-16 | One test execution | Claude proposal |
| P-17 | Search space `{lr}×{epochs}`, 4 trials, validation only | Claude proposal |
| P-18 | Checkpoint selection / tie-break / no early stopping | Claude proposal |
| P-19 | Duplication audit design, report-only policy | Claude proposal |
| P-20 | ADR-014 gate notebook and GO criterion | Claude proposal |
| — | **Colab tier for Phase 3–5** | **USER DECISION REQUIRED** |

**Gate status:** ADR-005 adversarial review of this proposal — **PENDING**.
This document must not be promoted into the source-of-truth docs before that review is run,
its findings are persisted under `reviews/`, and each material finding is dispositioned.
