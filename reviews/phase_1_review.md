# Phase 1 Codex Review (native `/codex:review`)

- **Review date:** 2026-08-11
- **Reviewed phase:** Phase 1 — Environment Setup + Data Preparation
- **Reviewer:** Codex, via the native `/codex:review` command (`codex-companion.mjs`), run against the full branch diff `worktree-phase1-env-dataset` vs `main`
- **Relation to other Phase 1 review evidence:** This is a second, separate review pass, distinct from `reviews/phase_1_code_review.md` (which was produced earlier in the same session via the `codex:rescue` skill against the worktree state at that time, before some corrections had landed). This review targets the final committed branch diff.
- **Review outcome:** Processed. All four findings ACCEPTed and fixed; see disposition table and re-verification evidence below.

---

## Codex's own summary

The authoritative Colab environment notebook always aborts in its checked-in state and is not safe to rerun after a kernel restart. Dependency drift and a test-conversion assertion that cannot detect failures also undermine reproducibility and phase-gate correctness.

## Full review comments (verbatim)

- **[P1] Populate the Colab repository URL** — `notebooks/00_environment.ipynb:120`
  In every real Colab run, `IN_COLAB` is true while this checked-in value is empty, so the notebook raises before dependency, CUDA, or processor checks. The repository already has an `origin`, so requiring users to edit the notebook makes the pending Colab validation impossible via the required Restart/Run All workflow (AGENTS.md L776-L784); configure the repository URL and intended revision in the notebook.

- **[P2] Make repository setup idempotent across kernel restarts** — `notebooks/00_environment.ipynb:135-138`
  After one successful Colab run, restarting the kernel normally preserves `/content/repo`; the next Run All therefore makes `git clone ... repo` exit because the destination already exists, and the notebook raises before validation. This conflicts with the required restart-and-Run-All behavior (AGENTS.md L776-L784); reuse/update an existing checkout or recreate a known absolute checkout directory.

- **[P2] Pin the dependency versions used by Colab** — `pyproject.toml:11-16`
  On a fresh or future Colab runtime, every `>=` constraint can resolve to a different release, changing processor behavior or breaking compatibility between Torch, torchvision, Transformers, and datasets. Recording one local notebook's installed versions does not make later installation reproducible, and the experiment contract explicitly requires dependency versions to be pinned in `pyproject.toml` (docs/EXPERIMENT_SPEC.md L73-L79); use exact tested versions or a lock file consumed by the setup notebook.

- **[P2] Fail when any held-out row cannot be converted** — `notebooks/01_dataset.ipynb:581-583`
  If `convert_ground_truth` raises `ValueError` for one or more test rows, the exception is counted and this assertion still always succeeds because every completed iteration increments exactly one counter. The notebook can therefore exit successfully despite conversion failing on the held-out split, contradicting the pipeline condition that conversion complete without error across all splits (docs/EXPERIMENT_SPEC.md L85-L88); assert that `parse_failure_count == 0`.

---

## Claude Code disposition (2026-08-11)

| # | Finding | Priority | Disposition | Fix applied |
|---|---|---|---|---|
| 1 | Populate the Colab repository URL | P1 | **ACCEPT** | `notebooks/00_environment.ipynb`, cell `177b5cec`: `REPO_URL` set to the repo's actual GitHub remote (`https://github.com/Mr-Kondo/finetuning_vlm.git`), which now exists — it did not exist when this notebook was originally authored, so the placeholder was correct at the time; the finding is now actionable. |
| 2 | Make repository setup idempotent across kernel restarts | P2 | **ACCEPT** | Same cell: if `repo/` already exists on disk, `git -C repo pull` runs instead of `git clone`, so a second **Restart & Run All** on the same Colab VM no longer fails on "destination already exists." |
| 3 | Pin the dependency versions used by Colab | P2 | **ACCEPT** | `pyproject.toml`: all five runtime dependencies (`datasets`, `transformers`, `torch`, `pillow`, `torchvision`) changed from `>=` floors to exact `==` pins, using the versions verified working together in this session. This directly satisfies `docs/EXPERIMENT_SPEC.md` §7 ("Pin dependency package versions (pyproject.toml, created in Phase 1)"), which an earlier review pass (`reviews/phase_1_code_review.md` finding #9) under-weighted by treating this purely as a "lock file" ask; exact version pins require no new tooling, so the earlier partial rejection is superseded here. |
| 4 | Fail when any held-out row cannot be converted | P2 | **ACCEPT** | `notebooks/01_dataset.ipynb`, cell `74000e11`: added `assert parse_failure_count == 0` (the pre-existing `assert parse_success_count + parse_failure_count == total_rows` is kept, but it only checks internal counter consistency, not that zero failures occurred — that gap is what this finding correctly identified). |

## Re-verification performed

- Both notebooks re-executed top-to-bottom locally (`jupyter nbconvert --execute --inplace`), exit 0 for both, real (non-fabricated) outputs.
- `pytest tests/` → 18 passed, 1 skipped (unchanged from before this fix pass — these findings didn't touch `data.py`).
- `scripts/validate_notebooks.py` → both notebooks OK.
- Test-split blindness re-confirmed after re-execution: exactly 4 `image/png` outputs (2 train + 2 validation), 0 occurrences of a `DatasetDict({` repr leak.
- `pip install -e ".[dev]"` succeeds with the new exact-pinned dependency versions.

## Unresolved questions

None outstanding from this review. All four findings were actionable and have been fixed and re-verified.
