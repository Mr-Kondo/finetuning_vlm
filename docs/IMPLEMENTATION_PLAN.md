# IMPLEMENTATION_PLAN.md — Implementation Plan

**Status:** Draft — Codex adversarial review completed (Phase 0)
**Related:** [[EXPERIMENT_SPEC]] §8a (each pipeline-completion item corresponds to a phase gate here) / [[EVALUATION_PROTOCOL]] / [[STATE]] / [[DECISIONS]] ADR-005

**Premise:** this document is a deliverable of Phase 0 (design specification drafting), and **no code implementation is performed during Phase 0**. The following is the plan for Phase 1 onward. Starting Phase 1 requires a Codex adversarial review and user approval ([[DECISIONS]] ADR-005).

## 1. Repository Layout (Planned)

```
vlm-colab-lab/
│
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── pyproject.toml
│
├── docs/
│   ├── EXPERIMENT_SPEC.md       # Experiment specification (what & why)
│   ├── IMPLEMENTATION_PLAN.md   # This file (how to build it)
│   ├── EVALUATION_PROTOCOL.md   # Evaluation method (ensuring a fair comparison)
│   ├── DECISIONS.md             # ADR log
│   └── STATE.md                 # Current phase / progress
│
├── configs/
│   ├── qwen_cord_smoke.yaml     # For connectivity checks (tiny data, a few steps)
│   ├── qwen_cord_mini.yaml      # For fast iteration (validation only)
│   └── qwen_cord_full.yaml      # For the final comparison (full train set, test evaluation)
│
├── notebooks/
│   ├── 00_environment.ipynb     # Colab environment setup, dependency installation
│   ├── 01_dataset.ipynb         # CORD v2 loading, format-conversion verification
│   ├── 02_baseline.ipynb        # Held-out test evaluation of the Base model
│   ├── 03_qlora_smoke.ipynb     # QLoRA smoke test (pipeline connectivity check)
│   ├── 04_qlora_train.ipynb     # Full QLoRA training (mini → full)
│   └── 05_evaluation.ipynb      # Final Base vs Fine-tuned comparison
│
├── src/
│   └── vlm_lab/
│       ├── data.py              # CORD v2 loading, Donut format → instruction format conversion
│       ├── inference.py         # Shared Base/Fine-tuned inference wrapper
│       ├── training.py          # QLoRA training loop (Trainer configuration)
│       ├── evaluation.py        # Shared Base/Fine-tuned evaluation metric computation
│       └── utils.py             # Seed fixing, config loading, logging
│
├── tests/                       # Unit tests for src/vlm_lab
├── scripts/
│   └── validate_notebooks.py    # Notebook executability check
├── reviews/                     # Records of Codex adversarial reviews
└── results/                     # metrics.json / report.md per evaluation run
```

## 2. Phase List

[[STATE]] must always be updated when each phase completes. Before starting implementation between phases, if a non-trivial design decision is involved, a Codex review must be run again ([[DECISIONS]] ADR-005).

### Phase 1: Environment Setup + Data Preparation
- `00_environment.ipynb`: dependency installation on Colab, measuring actual GPU/VRAM, finalizing `pyproject.toml`.
- `01_dataset.ipynb`: implement `data.py`. Convert CORD v2's `ground_truth` (Donut format) into a clean JSON schema for instruction tuning. **Development, debugging, and visual inspection of the conversion logic are done using `train`/`validation` only. Once the conversion logic is finalized, it is applied mechanically to `test` exactly once, with only record count and parse success/failure checked automatically (visual inspection is prohibited — [[DECISIONS]] ADR-008).**
- Additional Phase 1 exit conditions (do not proceed to Phase 2 until all are met):
  - **LoRA approval gate ([[DECISIONS]] ADR-012):** confirm the Qwen3-VL implementation's module list against the live model, and finalize and record in an ADR the fully-qualified names of the target modules, the target tower (language side / vision side / projector), the adapter parameter count, the freeze status of vision/projector, and whether assistant-only label masking is used.
  - **VRAM go/no-go gate ([[DECISIONS]] ADR-014):** measure the actual peak VRAM after forward/backward + optimizer-state allocation, using the production-intended image size, sequence length, LoRA rank, and microbatch size. This is not substituted by a reduced configuration like the smoke test that only shrinks record count and step count. Since T4 (Turing) does not support the standard FlashAttention-2 kernels, an explicit fallback path such as SDPA must be prepared and measured as well.
  - **Duplication audit ([[DECISIONS]] ADR-008):** automatically audit image/ground-truth-structure duplication between train/test using image hashes/perceptual hashes, and record the results and the handling policy (exclusion / re-split / reporting only).
  - **Revision pinning ([[DECISIONS]] ADR-015):** pin the Hub revisions of the model, processor, dataset, and (if used) quantization artifact to commit SHAs.
  - **Completion of pre-registration ([[DECISIONS]] ADR-009):** finalize, before observing any performance output in Phase 2, the pre-registration elements from [[EXPERIMENT_SPEC]] §8b (primary metric formula, threshold X, prompt/shot design, parser/normalization specification, hyperparameter search space, checkpoint-selection rule, allowed number of test executions) and the normative schema/metric definitions from [[EVALUATION_PROTOCOL]] §5.1.
- Exit conditions: [[EXPERIMENT_SPEC]] §8a-1, 8a-2, the additional exit conditions above, and finalizing whichever of the remaining open items in §10 are resolvable (image resolution, LoRA rank/alpha/dropout values, etc.).

### Phase 2: Baseline Evaluation
- `02_baseline.ipynb`: initial implementation of `inference.py` / `evaluation.py`. Run inference with the Base model on a small subset of `train`/`validation` to verify pipeline connectivity. **The `test` images, ground truth, predictions, and aggregate metrics are not generated or displayed at all at this stage ([[DECISIONS]] ADR-008).**
- Exit condition: [[EXPERIMENT_SPEC]] §8a-3 (not including test).

### Phase 3: QLoRA Smoke Test
- `03_qlora_smoke.ipynb`: using `qwen_cord_smoke.yaml`, confirm that QLoRA training can run on a tiny subset with only a few steps. **Performance is not evaluated. Only confirm that it does not crash and that the adapter can be saved.** Only the record count and step count are reduced; image size, sequence length, rank, and microbatch size match the production-intended values used in Phase 1's VRAM go/no-go gate ([[DECISIONS]] ADR-014).
- Exit condition: [[EXPERIMENT_SPEC]] §8a-4.

### Phase 4: Full QLoRA Training
- `04_qlora_train.ipynb`: fast iteration with `qwen_cord_mini.yaml` (referencing only the validation split; test is not touched), followed by the final training run with `qwen_cord_full.yaml`.
- Exit conditions: [[EXPERIMENT_SPEC]] §8a-5, 8a-6.

### Phase 5: Evaluation
- `05_evaluation.ipynb`: following [[EVALUATION_PROTOCOL]], evaluate both Base and Fine-tuned on the entire held-out test set, and output metrics.json / report.md to `results/`.
- Exit condition: [[EXPERIMENT_SPEC]] §8a-7 (pipeline completion achieved). At this point, the §8b (model performance improvement) determination first becomes possible.

### Phase 6: Report and Retrospective
- Report the pipeline-completion and model-performance-improvement results independently.
- Even if there was no improvement or there was regression, perform and record a root-cause analysis ([[EXPERIMENT_SPEC]] §8b).

## 3. Roles of the Three `configs/` Stages

| config | Data scale | Purpose | Splits used |
|---|---|---|---|
| `qwen_cord_smoke.yaml` | Tiny (a few to a few dozen rows), a few steps | Pipeline connectivity check only. Performance is not evaluated | train subset only |
| `qwen_cord_mini.yaml` | Small to medium, fast iteration | Prompt/hyperparameter tuning, early bug detection | train subset + validation |
| `qwen_cord_full.yaml` | Full 800-row train set, planned epoch count | Full training for the final comparison | full train set (validation only for tuning; test is not allowed) |

## 4. `src/vlm_lab` Module Responsibilities

- **data.py**: loading CORD v2 and converting from Donut format to instruction format. The correctness of this conversion directly determines the correctness of the ground truth.
- **inference.py**: model loading (base / with LoRA adapter applied) and a generation wrapper. **Both Base and Fine-tuned evaluation must call the same function** ([[DECISIONS]] ADR-006).
- **training.py**: the QLoRA training loop (PEFT Trainer configuration, config-driven hyperparameters).
- **evaluation.py**: computation of JSON validity / field-level F1 / TED-Acc / Exact Match. **Both Base and Fine-tuned evaluation must call the same function.**
- **utils.py**: seed fixing, config (yaml) loading, logging, and attaching artifact metadata (git commit hash, etc.).

## 5. `reviews/` and `results/`

- `reviews/`: stores the output of Codex adversarial reviews (date, target phase, findings, disposition status).
- `results/`: stores the metrics.json / report.md from [[EVALUATION_PROTOCOL]] §8, per run.

## 6. Phase-Gate Operating Rules

- Before moving into a phase that involves implementation (code changes), if there is a non-trivial design decision, a Codex adversarial review must be performed (also documented in [[CLAUDE]] / [[AGENTS]]).
- Review results are stored in `reviews/`, and the response plan for each finding is recorded in [[DECISIONS]] as an ADR.
- On completion of each phase, update the phase status and "next actions" in [[STATE]].
