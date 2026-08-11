# STATE.md — Current State

**Last updated:** 2026-08-11

This file must always reflect the latest state. Agents must always read this file before starting work in this repository ([[CLAUDE]] / [[AGENTS]]).

## Current Phase

**Phase 1: Environment Setup + Data Preparation — Implementation complete for this session's sub-scope; Colab validation FAILED on first real attempt (fixed, re-validation PENDING); additional Phase 1 exit conditions remain OPEN (see below).**

- Phase 0 is complete. The user approved the Phase 0 → Phase 1 transition explicitly; the ADR-005 adversarial-review gate for this transition was already satisfied by `reviews/phase_0_adversarial.md` (no new adversarial review was required to start Phase 1 — see that review and `reviews/phase_1_code_review.md` for the record).
- This session's Phase 1 task was explicitly scoped to: repository/environment bootstrap, `notebooks/00_environment.ipynb`, `notebooks/01_dataset.ipynb`, `src/vlm_lab/data.py`, and local tests. It did **not** cover every exit condition listed in `IMPLEMENTATION_PLAN.md`'s Phase 1 section — see "Unresolved Open Items" below for what remains.
- **Delivered and locally verified:**
  - `pyproject.toml` — installable `vlm_lab` package with exact-pinned runtime dependencies (`datasets==5.0.1`, `transformers==5.15.0`, `torch==2.13.0`, `pillow==12.3.0`, `torchvision==0.28.0`, per `EXPERIMENT_SPEC.md` §7's pinning requirement; dev extras `pytest`, `nbformat`).
  - `src/vlm_lab/data.py` — `load_cord_v2()` (loads all three Hub splits; real split sizes confirmed: train=800, validation=100, test=100) and `convert_ground_truth()` (normalizes Donut's list/dict-collapse quirk for `menu`/`void_menu`/nested `sub`; fails loudly with `ValueError` on any structurally malformed input).
  - `notebooks/00_environment.ipynb` — Colab-first environment/CUDA/version-reporting notebook; loads `Qwen/Qwen3-VL-4B-Instruct`'s config and processor (lightweight, no full model weights). The Colab-only repo-clone cell now points at the repo's real GitHub remote (`REPO_URL`) and is idempotent across kernel restarts (updates via `git pull` if `repo/` already exists instead of failing on re-clone). Executed locally: `cuda_available: False` (correct/expected on this Mac — Colab-authoritative result still pending).
  - `notebooks/01_dataset.ipynb` — CORD v2 inspection notebook. Displays real train/validation images + converted annotations; treats `test` mechanically only (row count + parse-success/failure counts, no content/images), per ADR-008. The mechanical test-check cell now asserts `parse_failure_count == 0` explicitly (a prior version's assertion only checked counter consistency, which is always true regardless of failures).
  - `tests/test_data.py` — 18 passing unit tests (+1 explicitly skipped network-gated integration test) covering `convert_ground_truth`'s normalization and malformed-input handling.
  - `scripts/validate_notebooks.py` — generic notebook structural-validation script (`nbformat` schema check + fabricated-output detection). Both notebooks pass.
  - `reviews/phase_1_code_review.md` — independent code review (via `codex:rescue`, this repo's stand-in for `/codex:review`; see project memory `delegation_tooling_mapping`). All required findings ACCEPTed-and-fixed or DEFERred with explicit rationale; none silently dropped.
  - `reviews/phase_1_review.md` — a second independent review pass (native `/codex:review` against the full branch diff). 4 findings, all ACCEPTed and fixed (Colab `REPO_URL`, idempotent clone, exact dependency pinning, the `parse_failure_count == 0` assertion above).
- **Local validation: PASS.** `pytest tests/` → 18 passed, 1 skipped. `scripts/validate_notebooks.py` → both notebooks OK. Both notebooks execute top-to-bottom locally with real, non-fabricated outputs (`jupyter nbconvert --execute`), re-verified after the `reviews/phase_1_review.md` fix pass.
- **Colab validation: PARTIAL — user attempted real Colab execution and reported it did NOT complete end-to-end as checked in.** Two concrete failures were reported (2026-08-11): (1) a plain `git clone` of `REPO_URL` checked out the default branch (`main`), which does not yet contain any of this Phase 1 work since PR #1 has not merged — the user had to manually `git pull origin <branch>` to get the actual files; (2) `notebooks/01_dataset.ipynb` had no environment-setup cell of its own and assumed `vlm_lab` was already installed (only true if `00_environment.ipynb` had already run in the *same* Colab runtime), so opening it fresh failed on the first `import vlm_lab`-dependent cell without a manual `pip install`. Both are now fixed (see below) but **have not yet been re-validated on actual Colab** — do not treat Colab validation as passing until the user re-runs both notebooks as checked in, with no manual intervention.
  - Fix: both notebooks now pin `GIT_REF = "worktree-phase1-env-dataset"` (the PR branch) for the clone/checkout, with a `TODO` to reset to `""` once PR #1 merges to the default branch.
  - Fix: `01_dataset.ipynb` now has its own self-contained Colab-detection + repo-clone + install cells (mirroring `00_environment.ipynb`'s), so it no longer depends on another notebook having run first in the same runtime.
  - Both fixes re-verified locally only (`IN_COLAB=False` branch unaffected; the `IN_COLAB=True` branch cannot be exercised outside real Colab) — full pytest (18 passed/1 skipped), notebook structural validation, and test-split blindness all still pass after these changes.
- **PR:** [#1](https://github.com/Mr-Kondo/finetuning_vlm/pull/1) (draft, open), branch `worktree-phase1-env-dataset`. Should not merge until Colab validation lands.
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

1. **User re-runs Colab validation** on the fixed notebooks: `notebooks/00_environment.ipynb` then `notebooks/01_dataset.ipynb`, each **Restart & Run All**, on a fresh runtime (or after pulling the latest commit on branch `worktree-phase1-env-dataset`), with **no manual git/pip intervention** — the whole point of this fix pass is that manual intervention should no longer be necessary. Report back whether it now completes end-to-end, and the actual CUDA/GPU/split-size output.
2. Once Colab validation lands, update this file with the actual results (COLAB PASS/FAIL, real GPU identity, real CUDA version).
3. Once PR #1 merges, reset `GIT_REF = ""` in both notebooks' Colab-setup cells (currently pinned to the PR branch since the default branch doesn't have this work yet).
4. Before Phase 1 as a whole (per `IMPLEMENTATION_PLAN.md`) can be declared complete and Phase 2 can start, the remaining open Phase 1 exit conditions below must still be completed — they were not part of this session's task scope.

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
| 2026-08-11 | Committed, pushed branch `worktree-phase1-env-dataset`, opened draft PR #1. |
| 2026-08-11 | Ran a second independent review pass via native `/codex:review` against the branch diff (`reviews/phase_1_review.md`). 4 findings, all ACCEPTed: populated the Colab `REPO_URL` (a real GitHub remote now exists), made the Colab repo-clone cell idempotent across kernel restarts, switched `pyproject.toml` dependencies from `>=` floors to exact `==` pins (per `EXPERIMENT_SPEC.md` §7), and fixed a test-conversion assertion in `01_dataset.ipynb` that could not actually detect parse failures. All fixes re-verified: both notebooks re-executed top-to-bottom with real outputs, `pytest` still 18 passed/1 skipped, test-split blindness re-confirmed. |
| 2026-08-11 | **User attempted real Colab execution; it did not complete end-to-end as checked in** — required manual `git pull origin <branch>` (plain clone got `main`, which lacks this work since PR #1 hasn't merged) and a manual package install for `01_dataset.ipynb` (which had no setup cell of its own). Fixed: both notebooks now pin `GIT_REF` to the PR branch for clone/checkout (with a TODO to reset once merged), and `01_dataset.ipynb` now has its own self-contained Colab-detection + clone + install cells, independent of `00_environment.ipynb` having run first. Re-verified locally (pytest, notebook structural validation, test-blindness); **not yet re-verified on actual Colab** — Colab validation remains open until the user confirms the fixed notebooks run end-to-end with no manual steps. |
