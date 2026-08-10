# STATE.md — Current State

**Last updated:** 2026-08-11

This file must always reflect the latest state. Agents must always read this file before starting work in this repository ([[CLAUDE]] / [[AGENTS]]).

## Current Phase

**Phase 1: Environment Setup + Data Preparation — Implementation complete for this session's sub-scope; Colab validation PENDING; additional Phase 1 exit conditions remain OPEN (see below).**

- Phase 0 is complete. The user approved the Phase 0 → Phase 1 transition explicitly; the ADR-005 adversarial-review gate for this transition was already satisfied by `reviews/phase_0_adversarial.md` (no new adversarial review was required to start Phase 1 — see that review and `reviews/phase_1_code_review.md` for the record).
- This session's Phase 1 task was explicitly scoped to: repository/environment bootstrap, `notebooks/00_environment.ipynb`, `notebooks/01_dataset.ipynb`, `src/vlm_lab/data.py`, and local tests. It did **not** cover every exit condition listed in `IMPLEMENTATION_PLAN.md`'s Phase 1 section — see "Unresolved Open Items" below for what remains.
- **Delivered and locally verified:**
  - `pyproject.toml` — installable `vlm_lab` package (`datasets`, `transformers>=4.57.1`, `torch`, `pillow`, `torchvision`; dev extras `pytest`, `nbformat`).
  - `src/vlm_lab/data.py` — `load_cord_v2()` (loads all three Hub splits; real split sizes confirmed: train=800, validation=100, test=100) and `convert_ground_truth()` (normalizes Donut's list/dict-collapse quirk for `menu`/`void_menu`/nested `sub`; fails loudly with `ValueError` on any structurally malformed input).
  - `notebooks/00_environment.ipynb` — Colab-first environment/CUDA/version-reporting notebook; loads `Qwen/Qwen3-VL-4B-Instruct`'s config and processor (lightweight, no full model weights). Executed locally: `cuda_available: False` (correct/expected on this Mac — Colab-authoritative result still pending).
  - `notebooks/01_dataset.ipynb` — CORD v2 inspection notebook. Displays real train/validation images + converted annotations; treats `test` mechanically only (row count + parse-success/failure counts, no content/images), per ADR-008.
  - `tests/test_data.py` — 18 passing unit tests (+1 explicitly skipped network-gated integration test) covering `convert_ground_truth`'s normalization and malformed-input handling.
  - `scripts/validate_notebooks.py` — generic notebook structural-validation script (`nbformat` schema check + fabricated-output detection). Both notebooks pass.
  - `reviews/phase_1_code_review.md` — independent code review (via `codex:rescue`, this repo's stand-in for `/codex:review`; see project memory `delegation_tooling_mapping`). All required findings ACCEPTed-and-fixed or DEFERred with explicit rationale; none silently dropped.
- **Local validation: PASS.** `pytest tests/` → 18 passed, 1 skipped. `scripts/validate_notebooks.py` → both notebooks OK. Both notebooks execute top-to-bottom locally with real, non-fabricated outputs (`jupyter nbconvert --execute`).
- **Colab validation: PENDING.** No GPU-dependent execution (real CUDA=True, real GPU identity, the Colab-only repo-clone branch) has actually run on Colab yet. Local execution proves structural correctness only, not GPU behavior — see "Colab Handoff" below (in the end-of-task report delivered to the user; repeat instructions here if this file is read without that context: open `notebooks/00_environment.ipynb` first, Restart & Run All, then `notebooks/01_dataset.ipynb`, Restart & Run All).
- **Pipeline completion vs. model performance:** not yet applicable — Phase 1 has no model-performance dimension (see [[EXPERIMENT_SPEC]] §8a/§8b). Nothing in this phase's scope contributes to that determination.

## Phase List and Progress

| Phase | Content | Status |
|---|---|---|
| 0 | Design specification drafting | Complete |
| 1 | Environment setup + data preparation | This session's sub-scope implemented + locally validated; Colab validation pending; additional exit conditions open (see below) |
| 2 | Baseline evaluation | Not started |
| 3 | QLoRA smoke test | Not started |
| 4 | Full QLoRA training | Not started |
| 5 | Evaluation (Base vs Fine-tuned) | Not started |
| 6 | Report / retrospective | Not started |

## Next Actions

1. **User runs Colab validation**: `notebooks/00_environment.ipynb` then `notebooks/01_dataset.ipynb`, each Restart & Run All, and reports back the results (see the Colab handoff instructions given at the end of this implementation session).
2. Once Colab validation lands, update this file with the actual results (COLAB PASS/FAIL, real GPU identity, real CUDA version).
3. Before Phase 1 as a whole (per `IMPLEMENTATION_PLAN.md`) can be declared complete and Phase 2 can start, the remaining open Phase 1 exit conditions below must still be completed — they were not part of this session's task scope.

## Unresolved Open Items (Blocking / Open)

See [[EXPERIMENT_SPEC]] §10. The following remain open and were **not** addressed by this session's implementation (scoped out per the user's explicit task boundary for this session, not overlooked — see `reviews/phase_1_code_review.md` findings #2 and #6 for the disposition):
- Fully-qualified LoRA target module names, target tower, and label masking ([[DECISIONS]] ADR-012, Phase 1 approval gate)
- Finalized prompt template, shot design, and demo-selection procedure ([[DECISIONS]] ADR-009)
- Image resolution and token budget
- Primary metric, improvement threshold X, and paired-bootstrap configuration ([[EVALUATION_PROTOCOL]] §6, §5.1, [[DECISIONS]] ADR-009, ADR-010)
- Production-shape VRAM go/no-go gate measurement ([[DECISIONS]] ADR-014)
- Train/test duplication audit ([[DECISIONS]] ADR-008) — not yet implemented; `data.py` has no hashing/audit logic yet
- Pinning of model/dataset/processor revisions (commit SHA) ([[DECISIONS]] ADR-015) — `load_cord_v2(revision=...)` accepts a revision but defaults to `None` (Hub default); no SHA has been chosen yet
- The Colab tier that will actually be used
- Actual Colab execution of `notebooks/00_environment.ipynb` and `notebooks/01_dataset.ipynb` (this session's new open item — implementation is done, execution is not)

## Change Log

| Date | Change |
|---|---|
| 2026-08-10 | Phase 0 started. Created seven new design docs. Confirmed the existence of the model/dataset. |
| 2026-08-10 | Conducted a Codex adversarial review (`reviews/phase_0_adversarial.md`). Of 12 findings, 11 were ACCEPTed; added ADR-008 through ADR-015; revised [[EXPERIMENT_SPEC]] / [[EVALUATION_PROTOCOL]] / [[IMPLEMENTATION_PLAN]]. The remaining item (multiple seeds) was held as USER DECISION REQUIRED. |
| 2026-08-10 | The user decided to adopt single-seed training (ADR-016). All 12 findings have now been addressed, and the design-level blockers for Phase 0 are resolved. |
| 2026-08-10 | Translated all docs/ files into English at the user's request. |
| 2026-08-11 | User approved the Phase 0 → Phase 1 transition and requested Phase 1 (environment + dataset prep) only. ADR-005 gate re-confirmed already satisfied (no new adversarial review needed). |
| 2026-08-11 | Implemented Phase 1 sub-scope: `pyproject.toml`, `src/vlm_lab/data.py`, `notebooks/00_environment.ipynb`, `notebooks/01_dataset.ipynb`, `tests/test_data.py`, `scripts/validate_notebooks.py` — delegated to general-purpose Agent (standing in for "GPT-5.6 Luna"), integration-verified by Claude Code at each step. |
| 2026-08-11 | Independent code review via `codex:rescue` (`reviews/phase_1_code_review.md`). 10 findings; required ones (malformed-input handling in `convert_ground_truth`, a minor test-blindness precaution, Colab-setup robustness, `transformers` version floor) fixed and re-verified; 2 findings (duplication audit, revision pinning) DEFERred as out of this session's scope with documented rationale. |
| 2026-08-11 | Local validation complete: 18/19 tests pass (1 correctly skipped), both notebooks structurally valid and execute top-to-bottom locally with real outputs. Colab validation still PENDING. |
