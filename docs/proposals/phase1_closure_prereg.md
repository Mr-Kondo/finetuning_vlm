# Phase 1 Closure and Phase 2 Pre-Registration — PROPOSAL (DRAFT v3.1)

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

**Governing principle from v3 onward: freeze the RULE, defer only the VALUE.** The round-2 review
correctly distinguished a legitimately deferred *measurement* (e.g. the exact p95 image-token count,
which needs a GPU) from a *pre-registration hole* (a decision rule that could still be chosen after
results are seen). Everywhere below, the rule, algorithm, formula, threshold and tie-break are fixed
here; only quantities that require execution are marked `TBD-MEASURED`, and each names the deliverable
that produces it.

**v2 → v3.** v2 was re-reviewed on 2026-08-12/13 (`reviews/phase_2_adversarial.md`, Round 2). Verdict
again **BLOCK**, with round-1 closure **CLOSED 11 / PARTIAL 12 / NOT CLOSED 2** and **13 new
`required`** findings, all dispositioned ACCEPT. v3 resolves them, cited inline as `[R2-n]`. Two
round-2 findings exposed real defects in how ADR-019 was drafted — near-duplicate grouping does not
remove selective training leakage, and excluding rows changes the estimand — now fixed by **ADR-021**.

**v1 → v2.** v1 was reviewed adversarially on 2026-08-12. Verdict **BLOCK**: 25 findings (17
`required`), all dispositioned **ACCEPT**. v2 resolved them and incorporated the four user decisions
recorded as ADR-017 through ADR-020. Findings cited inline as `[F<n>]`.

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
| ADR-019 (amended by ADR-021) | Cross-split duplicates **excluded — exact *and* near, per relation**; clustering reserved for **within-test** dependence; policy frozen before the audit runs |
| ADR-020 (amended by ADR-022) | Estimand is **trimmed verbatim transcription**; TED-Acc stays primary, with a mandatory per-receipt **index-free field-value multiset F1 diagnostic** |

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

# Step 2 — measured on the JOINT sequence, never by summing independent maxima [R2-11].
# Four distinct quantities, defined so they cannot double-count each other:
#   eval_prefix_len   = final processor output input_ids.shape[-1] for the evaluation message.
#                       This ALREADY contains the expanded image placeholder tokens.
#   train_seq_len     = input_ids.shape[-1] after constructing prefix + assistant target + EOS.
#   assistant_label_n = count of positions with label != -100 in that training sequence.
#   image_token_n     = prod(image_grid_thw) / merge_size**2, reported separately for the VRAM
#                       gate only — NOT an addend in the budget below [F18].
eval_prefix_len:    {p50: TBD-MEASURED, p95: TBD-MEASURED, max: TBD-MEASURED}
train_seq_len:      {p50: TBD-MEASURED, p95: TBD-MEASURED, max: TBD-MEASURED}
assistant_label_n:  {p50: TBD-MEASURED, p95: TBD-MEASURED, max: TBD-MEASURED}
image_token_n:      {p50: TBD-MEASURED, p95: TBD-MEASURED, max: TBD-MEASURED}

# Step 3 — derived by fixed formula from the joint measurements above
max_new_tokens: ceil(assistant_label_n.max * 1.25)
max_seq_len:    max(train_seq_len.max, eval_prefix_len.max + max_new_tokens)
```

**Assertion:** `max_seq_len` must not exceed the pinned model's maximum context; the gate fails loudly
if it does. `TBD-MEASURED` values come from §9 deliverable 2 and are frozen into `configs/*.yaml`
before Phase 2; the *formulas* above are frozen now and may not change after any result is seen.

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

### 5.3 P-10 — The template (ADR-020 as amended by ADR-022: trimmed verbatim)

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

Transcribe every value exactly as printed on the receipt. Preserve digit grouping such as
"58,000" or "23.000", currency symbols, capitalization, and internal spacing exactly as shown.
Do not convert, round, reformat, translate, or tidy any value. Leading and trailing spaces around
a value do not matter. Do not invent fields that are not visible on the receipt.

Return only the JSON object.
```

**`[R2-5]` The estimand is "trimmed verbatim transcription", not literal character-for-character.**
v2's prompt demanded spacing "character for character" while scoring stripped leading/trailing
whitespace — and vendored Donut's own `normalize_dict` strips scalar strings and drops empty values,
so literal character-for-character scoring is not achievable through the pinned evaluator at all. The
prompt above is corrected to match what is actually scored, and the estimand is renamed accordingly
throughout. ADR-020's substance is unchanged; only its name and this wording were inaccurate.

**Empty-string treatment, frozen `[R2-5]`:** a leaf whose value is `""` after trimming is treated as
**absent**, on both the prediction and reference sides, matching the vendored evaluator's behaviour.
This is stated here rather than inherited silently.

The key inventory is provisional and must be verified mechanically against **train + validation only**
(never `test` — ADR-008); if any key outside it occurs, the template is corrected before the
pre-registration is frozen (§9 item 1).

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

**Exact training-sequence construction, frozen `[R2-10]`:** v2 described it in prose ("same prefix,
then appends the assistant turn + EOS"), which is not an algorithm. The frozen procedure is:

1. Build the message list `[system, user(image, text)]` and render it with `add_generation_prompt=true`
   to obtain the **evaluation prefix**; process it to get `prefix_features`.
2. Build `[system, user(image, text), assistant(target_json)]` and render with
   `add_generation_prompt=false` to get `full_features`. **No EOS is appended manually `[R3-2]`** —
   the pinned Qwen chat template already emits `<|im_end|>\n` after the assistant content, and
   `<|im_end|>` *is* the EOS token. v3's earlier wording appended another, producing
   `target + EOS + EOS` while claiming "target plus EOS". The template's single terminator is the
   registered one.
3. `labels = full_features.input_ids.clone()`; set `labels[: prefix_features.input_ids.shape[-1]] = -100`.
   Every position at or before the prefix boundary — which includes all image-token positions — is
   masked, and only the assistant target plus its single terminator carries loss (§3.5).
4. Assert `full_features.input_ids[: prefix_len] == prefix_features.input_ids` element-wise, so the
   prefix boundary is verified rather than assumed.
5. **Assert exactly one terminal EOS**: `input_ids[-1] == eos_token_id` and `input_ids[:-1]` contains
   no `eos_token_id` beyond those the template legitimately emits for the system/user turns (counted
   from the prefix, so the assistant turn contributes exactly one).

**The canonical call is frozen, not conditional `[R3-2]`.** Both steps use the processor's own
`apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt")`. v3's earlier
"if the equivalence test fails, the processor's path wins" left the actual rule open; there is now
one path, and the equivalence test is retained only as a regression check that the rendered text and
the tokenized output agree.

**Input-equality guarantee `[F1]` `[F22]` `[R2-10]`:** the two conditions **share one precomputed
input object** — the features are built once per sample and passed to both, so divergence is
impossible by construction rather than detected after the fact. As a belt-and-braces check, assert
equality of the **complete `BatchFeature`**: the full key set, and per key the shape, dtype and a
content hash — explicitly including `pixel_values` and any position-related fields, not just
`input_ids` / `attention_mask` / `image_grid_thw` (two preprocessors can yield the same grid shape
with different normalized pixel tensors). Also assert structurally identical resolved
`GenerationConfig` and processor configuration.

The **only** permitted difference is the adapter state. v1's "no condition-dependent branch anywhere"
was an impossible criterion `[F22]`; the accurate rule is a one-item allowlist plus a structured diff
of everything else. Implementation: one quantized base instance with the PEFT adapter
enabled/disabled.

---

## 6. EVALUATION_PROTOCOL §5.1 — normative metric definitions

### 6.1 P-11 — Metrics (ADR-020)

- **Primary:** TED-Acc, per receipt. Sole input to the ADR-017 decision rule.
- **Mandatory diagnostic:** per-receipt **field-exact score**, defined below, reported with its own
  paired cluster CI. `[R2-4]` v2 called this a "guardrail", but ADR-020 deliberately makes it
  non-binding — a metric that cannot block a TED-Acc pass is a diagnostic, and calling it a guardrail
  overstated it. Its job is to make TED-Acc's insensitivity to single-character amount errors visible
  (`58,000` vs `59,000` costs one character edit but is a materially wrong amount) `[F8]`.
- Also reported: raw JSON validity, recoverable-payload parse rate, strict schema validity, micro-F1,
  Exact Match.

**P-11a — field-exact score, fully defined `[R2-4]`.** For one receipt:

1. Flatten both the prediction and the reference into a **multiset** of `(path, value)` pairs, where
   `path` is the dotted key path with **array indices replaced by `[]`** (so `menu[0].nm` and
   `menu[3].nm` share the path `menu[].nm`). Using a multiset with index-free paths makes the score
   invariant to menu-row ordering while still counting multiplicity — two identical rows contribute
   two pairs.
2. Apply §6.3's trimmed-verbatim normalization to `value`; drop pairs whose value is empty after
   trimming (consistent with §5.3's empty-string rule).
3. `TP = |pred ∩ ref|` as a **multiset intersection** (so duplicate rows must be duplicated correctly
   to earn credit); `FP = |pred| − TP`; `FN = |ref| − TP`.
4. `field_exact_i = 2·TP / (2·TP + FP + FN)`, i.e. per-receipt F1 over exact pairs. Defined as `1.0`
   when both multisets are empty, and `0.0` when exactly one is empty or the output failed to parse.

This is per-receipt and additive, so it has a well-defined paired `Δ_i` and its own CI under the §6.4
bootstrap — unlike corpus micro-F1 `[F12]`.

**P-11b — synthetic-error probe: characterization only, with a defined consequence `[R2-4]`.** Before
any model output exists, both metrics are scored on constructed cases — wrong amount digit, missing
field, extra field, item reorder, invalid JSON, near-correct OCR — and the table is transcribed into
`EVALUATION_PROTOCOL.md` §5.1.

Its status is **characterization, not a selection procedure**: it documents how each metric responds,
and it **cannot** change the primary metric on its own, because ADR-020 already fixed that and a
metric chosen from probe behaviour after the fact would be a post-hoc selection. The single defined
consequence: if the probe shows TED-Acc assigning **≥ 0.95** to a case with a wrong amount digit, that
fact is recorded prominently in `EVALUATION_PROTOCOL.md` §5.1 and in the Phase 6 report as a stated
limitation of the primary metric, and a change of primary metric is escalated as a **new
`USER DECISION REQUIRED`** — resolved before Phase 2 runs, never after results are seen.

### 6.2 P-12 — Donut evaluator, vendored and pinned `[F11]`

Vendor `JSONParseEvaluator` from `clovaai/donut` (`donut/util.py`) into
`src/vlm_lab/third_party/donut_eval.py` at a pinned commit SHA, with its license.

v1 claimed that "vendoring at a pin means there are no differences to document." That is **false**:
§6.3 wraps the evaluator in our own extraction, normalization and invalid-output rules, making the
whole thing a custom pipeline whose differences must be documented. Requirements:

1. **Pinned now, not deferred `[R3-9]`** — these are choices, not measurements, so leaving them as §9
   deliverables left metric behaviour selectable:

   ```yaml
   donut_repo: https://github.com/clovaai/donut
   donut_commit: 4cfcf972560e1a0f26eb3e294c8fc88a0d336626   # master as of 2026-08-13
   zss:  "1.2.0"
   nltk: "3.10.3"
   ```

   The SHA is recorded in `EVALUATION_PROTOCOL.md` §5.1 and in every result artifact.
2. **Transcribed from the pinned source, 2026-08-13** — read directly from
   `donut/util.py@4cfcf97`, not from memory. This closes the transcription half of `[F11]`:

   - **`normalize_dict`** sorts dict keys by `(len(key), key)`, so key **order is normalized away**;
     wraps every scalar value in a one-element **list**; and calls `str(value).strip()` on scalars —
     so it **stringifies** and **trims** every leaf. It also **drops falsy values**: `if not data:
     return {}` and `if value:` before assignment, so empty strings, empty dicts and empty lists
     disappear on both sides.
   - **`flatten`** emits `(dotted_path, value)` pairs in which **array indices are absent** — a
     two-row `menu` yields `("menu.nm", "A"), ("menu.price", "1"), ("menu.nm", "B"), …`.
   - **`cal_f1`** is a global micro-F1 over those pairs, matched as a **multiset**
     (`answer.remove(field)` on each hit), returning `TP / (TP + (FP+FN)/2)` — algebraically
     `2·TP / (2·TP + FP + FN)`.
   - **`cal_acc`** builds a tree (`<root>` → keys → `<subtree>` per list item → `<leaf>`-prefixed
     values), computes `zss.distance` with character `edit_distance` between leaf labels, and
     normalizes by **`TED(tree_of_empty_prediction, answer)`** — *not* by the answer's node count —
     then returns `max(0, 1 − nTED)`, so TED-Acc is bounded to `[0, 1]`.

   **Four consequences that change or confirm this document:**

   1. **The trimmed-verbatim estimand is correct, not a compromise** (ADR-022). `normalize_dict`
      strips every scalar, so literal character-for-character scoring is unreachable through this
      evaluator. §5.3's prompt now matches what is computed.
   2. **§5.3's empty-string rule is confirmed by the source**, not merely asserted: falsy leaves are
      dropped on both sides, which is exactly "empty counts as absent".
   3. **Donut's own F1 is itself index-free**, so `[R3-8]`'s association-blindness criticism applies to
      the official secondary metric too, not only to §6.1's companion metric. Both are reported with
      that limitation stated.
   4. **A numeric leaf still matches its string counterpart on content**, because `normalize_dict`
      stringifies. This is precisely why §6.5 reports **strict schema validity separately** — content
      credit and schema conformance genuinely diverge here.

   **Two operational details that must be pinned in the wrapper:**

   - `donut/util.py` imports `torch`, `datasets` and `transformers` at module level (for the
     unrelated `DonutDataset` class in the same file). Vendoring the file wholesale would drag the
     entire training stack into evaluation, so **only the `JSONParseEvaluator` class is extracted**,
     with its `zss` / `nltk` / `json` imports and the upstream MIT license header.
   - `cal_acc`'s denominator is `TED(empty, answer)`, which is **zero when `answer` is empty**,
     raising `ZeroDivisionError`. No CORD receipt is empty, but the wrapper must handle it explicitly
     rather than crash: an empty reference scores `1.0` if the prediction is also empty and `0.0`
     otherwise, and the occurrence is recorded.
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
| String normalization (ADR-020, **trimmed verbatim**) | **Strip leading/trailing whitespace only.** No NFKC, no case folding, no internal-whitespace collapse, no digit-grouping or currency normalization. v1's NFKC + collapse contradicted the verbatim instruction `[F9]`; v2's prompt still overclaimed "character for character" while scoring trimmed `[R2-5]`. Both prompt and estimand name are now aligned to what is actually computed. |
| Empty values `[R2-5]` | A leaf that is `""` after trimming counts as **absent** on both sides, matching the vendored evaluator's `normalize_dict`. Frozen here rather than inherited silently. |
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
bootstrap_unit: within_test_cluster   # see the algorithm below
bootstrap_weighting: receipt_weighted
min_retained_receipts: 60             # ADR-021 NOT EVALUABLE floor
min_independent_clusters: 40          # ADR-021 NOT EVALUABLE floor
```

**Cluster construction algorithm, fully specified `[R2-3]`.** v2 said "dHash ≤ 3, forming clusters",
which is not reproducible: Hamming adjacency is not transitive, so connected-components and
complete-linkage disagree, and it was unstated which signals contribute edges.

1. **Vertices:** the receipts of the **retained test set** — i.e. *after* ADR-021's cross-split
   exclusions have been applied. Clustering is therefore about residual **within-test** dependence
   only; cross-split leakage is handled by exclusion, never by clustering `[R2-1]`.
2. **Edges:** an undirected edge joins two test receipts when **either** their decoded-pixel hashes
   are equal (§8.1), **or** their dHash Hamming distance is `≤ 3`, **or** their value-inclusive
   canonical ground-truth hashes are equal. The **type-only template signature contributes no edges**
   — it would collapse many independent receipts into a few enormous clusters and is reported
   separately as template evidence only `[R2-3]`.
3. **Linkage:** **connected components** of that graph. Chosen over complete-linkage because it is
   deterministic, order-independent, and conservative (it never splits a genuinely dependent pair).
4. **Ordering:** exclusions are applied first, then the graph is built **once** over the retained set
   and never recomputed.
5. **Effective sample size:** `ESS = (Σ_c n_c)² / Σ_c n_c²` over cluster sizes `n_c` (Kish's formula).
   Reported alongside the retained receipt count. Purely descriptive — the decision rule uses the CI.

**Replicate algorithm, fully specified `[R2-3]`.** For each of `B = 10000` replicates: draw `K`
clusters **with replacement**, where `K` is the number of clusters in the retained set; concatenate
all receipts of the drawn clusters (so a large cluster contributes all its rows every time it is
drawn); compute the **receipt-weighted** mean of `Δ_i` over that concatenation. Receipt-weighted
rather than cluster-weighted so the statistic estimates the same quantity as the point estimate, which
is the plain mean over retained receipts. The 95% percentile interval is taken over the `B` means.

If every cluster is a singleton, this degenerates exactly to the ordinary paired receipt bootstrap;
that evidence is recorded either way `[F13]`.

**`NOT EVALUABLE` floor (ADR-021):** if the retained set has fewer than 60 receipts or fewer than 40
clusters, the confirmatory comparison is declared `NOT EVALUABLE` and no improvement claim is made.
Fixed before the audit so it cannot be chosen once the counts are known.

**Percentile over BCa `[F24]`:** chosen for transparency and implementation simplicity — one fewer
thing that can be silently wrong in the statistic the conclusion hinges on. v1's claim that BCa's
correction is "second-order at n=100" was unsupported and is withdrawn; sample size alone does not
establish negligible bias/skew correction for a bounded, possibly zero-inflated Δ distribution. The
trade-off is recorded honestly as a simplicity choice.

### 6.5 P-14b — Normative output schema `[F7]`

A key inventory is not a schema. `[R2-6]` v2 also contradicted itself — "unknown keys permitted
structurally" and "unknown keys are a schema violation" cannot both hold in one JSON Schema. The fix
is that there are **two separate paths**, and each is unambiguous:

| Path | Behaviour on an unknown key |
|---|---|
| **Strict schema validation** (`additionalProperties: false`) | invalid; contributes to the strict-validity rate |
| **Content metrics** (TED-Acc, field-exact, micro-F1) | tolerated structurally; the unknown pair simply counts as a false positive |

Strict validity is *reported*, never a gate on content scoring — so a parseable but schema-invalid
object still receives content credit, and that fact is now explicit rather than emergent.

**Deterministic construction algorithm, frozen now; the schema document itself is generated by it
`[R2-6]`.** Run over the **union of train + validation** converted ground truths (never `test`):

1. **Draft:** JSON Schema **2020-12**. Validator: `jsonschema==4.26.0` (§7.5), recorded in the
   artifact.
2. **Root:** `type: object`, `additionalProperties: false`, `required: []` — every top-level key is
   optional, because §5.3's prompt instructs omission of absent keys. Every generated object node
   carries the same two attributes.
3. **Property set at each level:** exactly the keys observed at that path in the corpus. No key is
   added by judgement and none is dropped for rarity. **Keys are emitted sorted**, so the document
   depends only on the observed key *set*, never on corpus iteration order.
4. **Shape at each path** — exhaustive, with no fall-through:
   - every observed value is a string → `{"type": "string"}`;
   - every observed value is an object → recurse as a strict object node;
   - every observed value is an array → `{"type": "array", "items": <recursed object node built from
     all elements>}`;
   - **anything else raises and fails the gate**, naming the path and every shape seen. "Anything
     else" includes: mixed shapes at one path, an array containing a non-object, `null`, and any
     non-string scalar.

   > **`[R3-7]` — corrected.** v3 ended this step with "Otherwise → leaf", which silently degraded
   > any mixed or unsupported shape into a string leaf and would therefore emit a schema that
   > **rejects its own reference corpus**. Silent degradation is replaced by a hard gate failure: an
   > unencodable corpus shape is a specification revision to be made before Phase 2, never something
   > absorbed at generation time.

5. **Leaf type:** `type: string`. All CORD leaves are strings after `convert_ground_truth`, so a
   numeric leaf in a *prediction* is a strict-schema violation — even though the vendored evaluator's
   `normalize_dict` stringifies it and may still let its value match on content (§6.2). That
   divergence is exactly why strict validity is reported separately.
6. **Nulls:** not permitted. `convert_ground_truth` never emits one, and if the corpus nevertheless
   contains a `null`, generation **fails** rather than emitting a nullable schema. In a *prediction* a
   `null` leaf is a strict violation and, per §6.3, is treated as *absent* for content scoring.
7. **Empty arrays:** permitted, and `minItems` is not set. A path observed **only** as empty arrays
   yields `{"type": "array"}` with **no `items`** — absence of evidence must not be converted into a
   constraint. Consequence, stated rather than hidden: such a path is *less* strict than the rest of
   the schema (an array of scalars would validate there). Every such path is listed in the
   generation report; `void_menu` is the realistic candidate, and whether it ever occurs non-empty in
   train+validation is unknown until §9 deliverable 1 runs.
8. **Empty corpus:** generation fails. A schema with no properties would silently reject every
   non-empty object.
9. The generated document is written to `configs/cord_v2_output.schema.json`, **hashed**, and the hash
   recorded in every result artifact. The hash is computed over a **canonical serialization of the
   parsed object** (sorted keys, compact separators), not over the committed file's bytes, so
   reformatting the file cannot change it. `$schema` is included in the hashed content, because a
   draft change is a meaning change.

**Validator fixtures, frozen `[R2-6]`:** the generated schema must reject — unknown key, numeric leaf,
`null` leaf, a bare object where an array is required — and must accept — a minimal single-`menu`-item
receipt, an empty array, and every train+validation reference. These fixtures are committed and run in
CI, so a later schema regeneration cannot silently change meaning.

### 6.6 P-15 — Seed

```yaml
seed: 42            # LoRA init, data ordering, any sampling; single seed per ADR-016
```

---

## 7. Decision rule, test budget, and the training configuration

### 7.1 Decision rule (ADR-017)

`X = 0.05`, on TED-Acc, against `CI_lower(Δ) ≥ X` from the §6.4 cluster bootstrap. Regression is
`CI_upper(Δ) < 0`. Field-exact is reported alongside as a **diagnostic** and is **not** part of the
decision rule (ADR-020, `[R2-4]`).

**Estimand (ADR-021, `[R2-2]`):** the confirmatory claim concerns **"CORD v2 test receipts with no
exact or near duplicate in train or validation"** — not the official 100-row split, since ADR-019's
exclusions are non-random and preferentially remove template-repetitive receipts. Every report must
state this. Results over the full official 100 rows may appear **only** as a clearly-labelled
non-confirmatory diagnostic. If the retained set falls below the §6.4 floor, the result is
`NOT EVALUABLE`. ADR-017's threshold value is unchanged; only its written "resolvable at n=100"
rationale was stale and is amended by ADR-021.

### 7.2 P-16 — Test-execution budget `[F6]`

**One** execution, in Phase 5, covering both conditions in the same run.

`[R2-13]` v2 said infrastructure failure was "the only condition permitting a rerun" and then
immediately permitted a corrected-config rerun for code defects — a contradiction that would allow
test-driven correction. The policy is now **four disjoint cases, distinguished by how much test
information has escaped**:

| # | Situation | Policy |
|---|---|---|
| 1 | **Infrastructure/artifact failure** (runtime disconnect, crash, corrupted or incomplete artifact) — no output observed | **Exact-config replay.** Nothing about the run's config or code may change. |
| 2 | **Startup defect before any test example is processed** (e.g. a config fails validation, the model fails to load) | Fix, then run. No test data has been touched, so this is not a test execution at all. Recorded in `docs/STATE.md`. |
| 3 | **Code/config defect discovered after test examples were processed but while outputs are still sealed** | The run is voided and the sealed outputs are **destroyed unread**. The defect, its evidence and the corrected config are recorded in `docs/STATE.md` *before* the rerun. |
| 4 | **Defect discovered after any output, metric or test-dependent failure detail has been observed** | The confirmatory claim is **invalidated**. A rerun does not restore it. Continuing requires a **new ADR and an explicit user decision**, and any subsequent result is reported as non-confirmatory. |

- **Generation-length termination** (outputs reaching `max_new_tokens`) is model behaviour, **not** a
  defect, and never grounds for a rerun or a larger cap — under any of the four cases.
- Outputs stay **sealed** until both conditions and the artifact-integrity checks have completed, so a
  Base-complete / Fine-tuned-failed run cannot expose Base test results during debugging.
- Every attempted test-split access is logged (§8.3).

### 7.3 P-17 — Search space and split roles `[F14]` `[F15]`

v1's "everything except the two knobs below is fixed" was false. The complete training configuration:

```yaml
# --- searched (the ONLY free parameters) ---
search_space:
  learning_rate: [1e-4, 2e-4]
  num_train_epochs: [2, 3]
max_trials: 4                        # full 2x2 grid; no adaptive search, no extra rounds

# --- fixed [F14] ---
# Trainer contract [R2-7]: transformers.Trainer with transformers.TrainingArguments from the
# pinned transformers==5.15.0, plus a project-owned multimodal collator. NOT SFTTrainer.
# Field names below are the pinned runtime's actual TrainingArguments names; v2 used
# `optimizer`, `dataloader_shuffle_seed` and the removed `group_by_length`, none of which exist.
# `packing` was also a different layer's concept and is dropped.
optim: paged_adamw_8bit              # valid in transformers 5.15; bitsandbytes supports it on T4
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
gradient_checkpointing_kwargs: {use_reentrant: false}
per_device_eval_batch_size: 1
data_seed: 42                        # sampler/data ordering
train_sampling_strategy: random      # [R3-10] the pinned runtime's default, stated explicitly
seed: 42
save_strategy: epoch
eval_strategy: epoch
save_total_limit: null               # keep every epoch checkpoint; selection is explicit (§7.4)
load_best_model_at_end: false        # selection is done explicitly, not by the trainer
remove_unused_columns: false         # REQUIRED: the collator consumes non-tensor image columns
fp16: true
bf16: false
report_to: []
```

**Generative validation, frozen `[R3-1]`.** `eval_strategy: epoch` alone invokes
`Trainer.evaluate()`, whose prediction step returns teacher-forced loss and logits — it does **not**
generate JSON, so it cannot produce the validation TED-Acc that §7.4 selects on, and it would also
retain vocabulary-sized logits for every token. The frozen path:

- A named **`TrainerCallback`** (`ValidationGenerationCallback` in `src/vlm_lab/training.py`) runs at
  `on_evaluate`, after the epoch checkpoint has been saved.
- It calls the **same shared inference and evaluation functions** used by Phase 5 (ADR-006), over the
  retained validation split, and writes `{epoch, checkpoint_path, ted_acc, field_exact}` to a
  selection log.
- The trainer's own loss-based evaluation is left enabled only for the training-loss curve;
  `metric_for_best_model` is **not** set and `load_best_model_at_end: false`, so the trainer never
  performs selection. Selection is done afterwards, mechanically, from the selection log per §7.4.

**Quantized-model preparation order, frozen `[R3-10]`:**

```
load 4-bit model (§2)
  → prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                    gradient_checkpointing_kwargs={"use_reentrant": False})
  → get_peft_model(model, LoraConfig(...))   # §3.2/§3.3
  → model.config.use_cache = False
```

This order matters: k-bit preparation changes frozen parameters, dtypes and gradient-input behaviour,
and doing it after adapter injection gives different results. The `use_reentrant: false` path is
validated against the pinned `peft` version in §8.2 arm A.

Additionally frozen outside `TrainingArguments` `[R2-7]`:

- **Collator:** a project-owned multimodal collator in `src/vlm_lab/training.py` that performs the
  §5.4 construction per sample and stacks a single-example batch. Named and unit-tested, not implicit.
- **`model.config.use_cache = False` during training** (mandatory with gradient checkpointing).
- **Config loading rejects unknown keys** — an unrecognized YAML key is an error, not a silent no-op,
  so a field name that does not exist in the pinned runtime cannot pass unnoticed (which is exactly
  how v2's three wrong names survived).
- The **fully resolved** `TrainingArguments` are serialized into the run artifact.

**Split roles, stated unambiguously `[F15]`:** `validation` is **evaluation-only and never updates
weights**; `test` is untouched until Phase 5.

**Trial and final-artifact provenance, frozen `[R2-8]`:** all four grid trials fit on the **full
800-row `train` split** using `qwen_cord_full`; the selected checkpoint is used directly as the
Fine-tuned artifact, with **no retraining after selection**. Consequently:

- `qwen_cord_mini`'s role is narrowed to **pre-grid smoke iteration only** (fast bug detection); it
  produces **no** artifact that can become the final model, and its results never feed selection.
  `IMPLEMENTATION_PLAN.md`'s configs table must be corrected accordingly, along with its existing
  self-contradiction about whether mini references validation or train+validation `[F15]`.
- v2 also justified "no retraining" by claiming a post-selection refit would "leak validation
  information". That was too broad and is withdrawn `[R2-8]` — validation-guided selection is the
  intended design. The actual reason is simpler: training all four trials on full train means the
  selected checkpoint is already trained on everything it would be retrained on, so a refit would add
  cost and a second seed-dependent artifact for no benefit.

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

- Add exact `==` pins. **Values fixed now `[R3-9]`**, since they are choices rather than measurements:

  ```toml
  peft         == 0.20.0
  bitsandbytes == 0.50.0
  accelerate   == 1.14.0
  jsonschema   == 4.26.0     # §6.5 strict-schema validator
  pyyaml       == 6.0.3
  zss          == 1.2.0      # §6.2
  nltk         == 3.10.3     # §6.2 — donut/util.py imports nltk.edit_distance
  ```

  Compatibility of these against the pinned `transformers==5.15.0` / `torch==2.13.0` / Python 3.12 is
  itself an execution check in the §8.2 gate arm A; a conflict there is a specification revision
  before Phase 2, not a silent version bump.
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

**Actions (ADR-021, superseding ADR-019's original wording) `[R2-1]`.** v2 excluded only *exact*
cross-split duplicates and demoted near matches to bootstrap clusters. That conflated two different
problems: clustering fixes dependence among evaluated receipts, but does nothing about **bias in the
point estimate** from the Fine-tuned model having trained on a near-copy the Base model never saw.
The action is therefore relation-specific:

| Relation | Exact match | Near match (dHash ≤ 3) |
|---|---|---|
| `train` ↔ `test` | excluded from test | **excluded from test** |
| `validation` ↔ `test` | excluded from test | **excluded from test** |
| `train` ↔ `validation` | excluded from validation | **excluded from validation** |
| `test` ↔ `test` | n/a | **retained** — the only relation the §6.4 cluster bootstrap handles |

Exclusion counts, the retained receipt and cluster counts, the effective sample size, and the realized
validation size are always reported — including when nothing is excluded. If the retained test set
falls below §6.4's floor, the result is `NOT EVALUABLE`.

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
| C — Resume | a genuine **save → destroy process → reload → one further optimizer step** cycle, covering adapter, optimizer state, scheduler, grad scaler, RNG state and dataloader position — not merely adapter save/reload `[R2-9]` |
| D — Eval | Base **and** Fine-tuned generation with the decode length **forced to the full configured `max_new_tokens`** (`min_new_tokens = max_new_tokens`, EOS suppressed) at maximum input length, so the KV cache actually reaches its worst case rather than stopping early at EOS `[R2-9]` |

**Measurement protocol, corrected `[R2-9]`.** v1 used an unsupported 13.0 GiB constant; v2 replaced it
with an invalid inequality — `max_memory_reserved()` is *this process's* allocator peak, while
`mem_get_info()` free memory is global and time-dependent, so comparing the peak against free memory
sampled later double-counts the process's own allocation. The protocol is now:

1. Each arm runs in a **fresh process**, so no earlier arm's allocator state carries over.
2. Immediately after CUDA init and before any model allocation, record
   `baseline_free, total = torch.cuda.mem_get_info()`. `baseline_free` is the budget for that arm.
3. Call `torch.cuda.reset_peak_memory_stats()` at the start of the measured region, and
   `torch.cuda.synchronize()` before reading any statistic.
4. Report `max_memory_allocated`, `max_memory_reserved`, and `total − min_free_observed`
   (a periodically sampled global low-water mark that also captures non-PyTorch allocations).
5. **GO criterion:** `max_memory_reserved ≤ baseline_free − 1.0 GiB` for every arm, where the 1.0 GiB
   is a **pre-registered fixed safety margin** for allocator fragmentation and driver overhead — a
   stated constant, not a residual inferred after the fact.
6. Because `paged_adamw_8bit` can silently fall back to CUDA unified-memory paging under pressure,
   arm B additionally reports per-step wall time; a step-time blow-up with apparently-fitting memory
   is recorded as a **soft NO-GO** `[R2-9]`.

Scale reference: fp32 Adam moments for 33,030,144 parameters are `33,030,144 × 2 × 4 = 252.0 MiB`
before gradients, master weights, activations and allocator effects — which is why the optimizer is
pinned in §7.3 rather than left unspecified.

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
depends on caller discipline. `[R2-12]` v2's "explicit allowed-splits argument" was no better: a
parameter the caller supplies is a parameter the caller can set to `"test"`.

**Three separately-named entry points, with capability baked into the function, not into an
argument:**

| API | Splits reachable | Used by |
|---|---|---|
| `load_development_splits()` | `train`, `validation` **only** — has *no parameter capable of naming test* | Phases 1–4, including `02_baseline.ipynb` |
| `load_for_duplication_audit()` | all three, but returns **hashes and counts only**, never images or ground-truth content (§8.1) | the ADR-008/ADR-019 audit |
| `load_sealed_test_split()` | `test` | Phase 5 only |

`load_for_duplication_audit()` and `load_sealed_test_split()` **log every invocation** (timestamp,
caller module, git commit) to an append-only access log committed with the results, so
`EXPERIMENT_SPEC.md` §8b's "one test execution" is auditable rather than asserted.

**Test asserts on the request, not the return value `[R2-12]`:** the unit test patches the loading
boundary and asserts that the *split argument actually sent to the Hub* from
`load_development_splits()` never contains `"test"` — checking the returned object would pass even if
test had already been downloaded.

### 8.4 P-22 — Per-sample evidence artifact `[F17]`

Aggregate `metrics.json` + `report.md` cannot support an audit of paired statistics: a wrong join, a
dropped parse failure, or a duplicated sample still yields plausible aggregates. Pre-register an
immutable per-sample record containing: stable sample ID, condition, raw output, generation
termination reason, raw-validity and recoverable-parse status, parsed object, reference linkage hash,
every per-sample metric, exception state, duplication-cluster ID, and config/artifact hashes.

**Assert one-to-one Base / Fine-tuned / reference alignment before bootstrapping.**

**Identity and immutability, specified `[R2-14]`** — v2 called the artifact "immutable" without saying
what made it so:

- **Format:** versioned **JSONL**, one record per `(condition, receipt)`, with a `schema_version` field
  and a header record naming it.
- **Unique key:** `(dataset_revision, split, row_index, condition)`. Asserted unique on write; the
  record count is asserted equal to `2 × retained_receipts`.
- **Write semantics:** written to a temporary path, then **content-addressed** — the file is hashed and
  renamed to include its hash, and the hash is recorded in `metrics.json`. Rewriting produces a
  different filename rather than mutating a file in place.
- **Exclusion manifest `[R2-14]`:** evaluated-row records alone cannot prove the *right* rows were
  excluded, so a separate sealed manifest lists every excluded row ID, the relation that caused it
  (`train↔test` exact, `validation↔test` near, …) and the cluster ID, plus its own hash. Sealed until
  Phase 5 completes; before then only aggregate counts are published (§8.1 test blindness).

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
| P-11 / P-11a / P-11b | TED-Acc primary + index-free field-value multiset F1 **diagnostic**; synthetic-error probe | proposal (ADR-020/022) |
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

**All four previously-open user decisions are closed** by ADR-017 through ADR-020, with ADR-021
amending ADR-019's write-up (not the decision) to close the leakage hole round 2 found.

### What v3 changed, by round-2 finding

| Finding | Resolution in v3 |
|---|---|
| R2-1 near-duplicate leakage | ADR-021: relation-specific exclusion table; clustering now covers *within-test* dependence only (§8.1, §6.4) |
| R2-2 estimand / `n=100` | ADR-021: estimand named explicitly; ADR-007 narrowed; `NOT EVALUABLE` floor; stale rationale amended (§7.1) |
| R2-3 cluster algorithm | Vertices, edge signals, connected-component linkage, ordering, ESS formula, replicate procedure, receipt weighting — all frozen (§6.4) |
| R2-4 field-exact | Full formula (index-free path multiset, multiset intersection, per-receipt F1); renamed a **diagnostic**; probe given characterization-only status with a defined escalation (§6.1) |
| R2-5 verbatim conflict | Estimand renamed **trimmed verbatim**; prompt aligned; empty-string rule frozen (§5.3, §6.3) |
| R2-6 JSON Schema | Two explicit paths for unknown keys; deterministic construction algorithm; draft/validator pinned; fixtures frozen (§6.5) |
| R2-7 training config | Real `transformers==5.15.0` field names (`optim`, `data_seed`); `group_by_length`/`packing` dropped; trainer, collator, `use_cache`, checkpoint semantics named; unknown YAML keys rejected (§7.3) |
| R2-8 provenance | All four trials on full `train`; mini narrowed to smoke iteration; the over-broad "refit leaks validation" claim withdrawn (§7.3) |
| R2-9 VRAM inequality | Fresh process per arm, `baseline_free` budget, peak reset, synchronized reads, fixed 1.0 GiB margin, real resume cycle, forced full decode, paging soft-NO-GO (§8.2) |
| R2-10 input freeze | One shared precomputed input object; full `BatchFeature` equality incl. `pixel_values`; exact training-sequence construction with a verified prefix boundary (§5.4) |
| R2-11 token budget | Four non-overlapping measured quantities; `max_seq_len` from joint sequence maxima, not summed components (§4) |
| R2-12 loader | Three capability-separated entry points; Phase 2's cannot name test; logged test access; test asserts on the request (§8.3) |
| R2-13 rerun policy | Four disjoint cases by how much test information escaped; case 4 invalidates the confirmatory claim (§7.2) |
| R2-14 artifact | Versioned JSONL, unique key, content-addressed writes, sealed exclusion manifest (§8.4) |
| R2-15 STATE | `docs/STATE.md` made internally consistent during this integration pass |

**Gate status:** ADR-005 adversarial review — **three rounds run, all BLOCK**
(`reviews/phase_2_adversarial.md`). Round 3 reviewed v3 and returned 15 further findings, all
dispositioned ACCEPT.

**v3.1** fixes the subset of round-3 findings that are unambiguous and self-contained: the double-EOS
training construction (R3-2), generative validation under a named callback instead of plain
`Trainer.evaluate()` (R3-1), the k-bit preparation order and `train_sampling_strategy` (R3-10), the
evaluator and dependency pins that were choices rather than measurements (R3-9), and the terminology
contradictions (R3-15, via ADR-022).

**Still open from round 3:** R3-3 (hidden-test input upper bound), R3-4 (one cross-split graph
algorithm), R3-5 (validation floor), R3-6 (ESS floor / stop implying resolvability), R3-7
(heterogeneous-shape gate failure), R3-11 (telemetry allowlist for rerun case 3), R3-12 (global
free-memory condition), R3-13 (timing threshold, arm D against an initialized adapter, two-process
resume), R3-14 (Phase-5-only sealed module + integration test), and R3-9's remaining item (freezing
the exact synthetic-probe fixtures).

This document must not be promoted, and `02_baseline.ipynb` must not be executed, until a review
passes.
