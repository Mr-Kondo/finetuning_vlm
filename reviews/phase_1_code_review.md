# Phase 1 Independent Code Review

- **Review date:** 2026-08-11
- **Reviewed phase:** Phase 1 — Environment Setup + Data Preparation (sub-scope: `notebooks/00_environment.ipynb`, `notebooks/01_dataset.ipynb`, `src/vlm_lab/data.py`, `tests/test_data.py`, `scripts/validate_notebooks.py`, `pyproject.toml`)
- **Reviewer:** Codex, via the `codex:rescue` skill (this repo's stand-in for `/codex:review` — see `reviews/phase_0_adversarial.md` note on tooling; confirmed with the user at the start of this session, see project memory `delegation_tooling_mapping`)
- **Review type:** Code review (AGENTS.md §9.1), not adversarial/design review — the ADR-005 adversarial-review gate for the Phase 0→1 transition was already satisfied by `reviews/phase_0_adversarial.md`.
- **Reviewed state:** worktree `phase1-env-dataset`, immediately after Tasks 1A–1E (delegated implementation) and the direct `torchvision` dependency fix.

## Findings and Claude Code disposition

Per `AGENTS.md` §10–12, every material finding is classified ACCEPT / REJECT / DEFER / USER DECISION REQUIRED with rationale. Findings 3 and the malformed-input claims embedded in Finding 3 were independently reproduced by Claude Code before disposition (not accepted on Codex's word alone).

| # | Finding | Priority (Codex) | Disposition | Rationale |
|---|---|---|---|---|
| 1a | `01_dataset.ipynb` cell 3 prints test split's row count outside the ADR-008 mechanical-only cell | required | **REJECT** | Row count alone is not ground-truth content or structure — it is already public information stated verbatim in `EXPERIMENT_SPEC.md` §3 ("test: 100 rows"). My own Task 1D delegation spec explicitly authorized confirming test's row count as an allowed mechanical check. Displaying already-pre-registered, non-content metadata does not create a test-blindness risk. |
| 1b | `01_dataset.ipynb` cell 2's bare `dataset` (`DatasetDict`) repr shows test's column names and row count | required | **ACCEPT** | Even though column names (`image`, `ground_truth`) are identical across all three splits and already documented, this cell is not something my task spec required, and removing it is free. Given ADR-008 is the single highest-stakes rule in this phase, precautionary removal costs nothing and removes any ambiguity for a future reader. |
| 2 | ADR-008 cross-split duplication audit is absent | required | **DEFER** | Valid and required by `IMPLEMENTATION_PLAN.md` Phase 1 exit conditions — but the user's task instructions for this specific session explicitly scoped Phase 1 deliverables to `00_environment.ipynb`, `01_dataset.ipynb`, `data.py`, and tests only (Section 3), and the Section 17 acceptance criteria for this task do not list the duplication audit. Not implementing it does not compromise anything already built. Tracked as an open Phase 1 exit condition in `docs/STATE.md`; must be completed before Phase 1 as a whole (per `IMPLEMENTATION_PLAN.md`) is declared done. |
| 3 | `convert_ground_truth` fails to validate structural types: `gt_parse` as a list silently returns `{}`; `menu` as a string is iterated into single characters; top-level/`gt_parse` `null` raises `TypeError` instead of `ValueError` | required | **ACCEPT** — independently reproduced, see below | Genuine correctness defects that violate the module's own documented "fail loudly on malformed input" contract. Dispatched as a bounded correction task. |
| 4 | Tests don't cover malformed-structure cases | required | **ACCEPT** | Follows directly from #3; regression tests bundled into the same correction task. |
| 5a | Colab setup cell doesn't check clone/install exit status or verify `import vlm_lab` after install | required | **ACCEPT** | Reasonable robustness fix, consistent with this repo's own "fail visibly when required operations fail" notebook rule. |
| 5b | Colab setup cell should fail explicitly if GPU is required but unavailable | required | **REJECT** | Would break the notebook's intentional, already-established design: honestly *report* CUDA availability (True/False) rather than crash when it's False. This same notebook is designed to run for structural verification on non-GPU environments (this local Mac, or an accidentally-CPU Colab runtime) without failing — later cells (config/processor load) do not need a GPU. Hard-failing on missing CUDA would make the notebook unusable for its documented local-structural-validation purpose. |
| 6 | Model/dataset/processor revisions are not pinned to commit SHAs (ADR-015) | required | **DEFER** | Same rationale as #2 — an explicit, separate Phase 1 exit condition not in this session's task scope. Also practically premature: `EXPERIMENT_SPEC.md` §2 states the choice between runtime quantization and a pre-quantized artifact is itself still an open Phase 1 decision, so there is no final revision to pin yet. Tracked as an open item in `docs/STATE.md`. |
| 7 | Checked-in `00_environment.ipynb` output has warning noise and an absolute local path | recommended | **ACCEPT** | Low-cost; addressed as a natural side effect of re-executing the notebook after the dependency and Colab-setup fixes below. |
| 8 | `transformers>=4.45` floor likely predates Qwen3-VL support | required | **ACCEPT** | Verified: PyPI release history shows `transformers` 4.45 (Sept 2024) predates Qwen3-VL's addition to the library by roughly a year. Corrected directly (see below). |
| 9 | No lock file / fully pinned dependency stack | required | **PARTIAL ACCEPT / REJECT lock-file mechanism** | The exact tested stack (Python 3.12.13, torch 2.13.0, transformers, datasets, torchvision, pillow versions) is already recorded as real, non-fabricated output inside `00_environment.ipynb` itself, satisfying the reproducibility intent. A separate lock-file toolchain (pip-compile/uv.lock) is new infrastructure this session's instructions explicitly caution against ("do not introduce unnecessary infrastructure") and is not requested by any Phase 1 acceptance criterion. Not adopted. |
| 10 | `docs/STATE.md` still says Phase 1 not started, `src/`/`notebooks/` don't exist | required | **ACCEPT** | Correct and expected — `docs/STATE.md` is updated as the final step of this task, after all corrections land (see below), not before. |

## Verification performed by Claude Code before disposition

Finding 3's claims were reproduced directly against the actual `src/vlm_lab/data.py` in the worktree (not taken on trust):

```
gt_parse is a list ('{"gt_parse": []}'): NO EXCEPTION, returned {}
menu is a string ('{"gt_parse":{"menu":"abc"}}'): NO EXCEPTION, returned {'menu': ['a', 'b', 'c']}
top-level null ('null'): TypeError - argument of type 'NoneType' is not iterable
gt_parse null ('{"gt_parse": null}'): TypeError - 'NoneType' object is not iterable
```

All four confirmed exactly as reported.

## Gate status

All required findings are either ACCEPTed-and-fixed, or DEFERred with an explicit, documented rationale tied to this session's task-scope boundary (not silently dropped). No finding was rejected without rationale. Corrections for findings 1b, 3, 4, 5a, 7, 8 were dispatched/applied after this review; see `docs/STATE.md` for the resulting status. Findings 2 and 6 remain open Phase 1 exit conditions and are recorded as such.
