# STATE.md — Current State

**Last updated:** 2026-08-10

This file must always reflect the latest state. Agents must always read this file before starting work in this repository ([[CLAUDE]] / [[AGENTS]]).

## Current Phase

**Phase 0: Design Specification Drafting — In Progress (review reflected, awaiting user approval)**

- CLAUDE.md / AGENTS.md / docs/EXPERIMENT_SPEC.md / docs/IMPLEMENTATION_PLAN.md / docs/EVALUATION_PROTOCOL.md / docs/STATE.md / docs/DECISIONS.md have been created.
- The existence and structure of the model (`Qwen/Qwen3-VL-4B-Instruct`) and dataset (`naver-clova-ix/cord-v2`) have been confirmed on the Hugging Face Hub (2026-08-10).
- **Codex adversarial review completed** (2026-08-10, `reviews/phase_0_adversarial.md`, confidence 0.94). Of the 12 material findings, 11 were ACCEPTed and reflected in `EXPERIMENT_SPEC.md` / `EVALUATION_PROTOCOL.md` / `IMPLEMENTATION_PLAN.md` / `DECISIONS.md` (ADR-008 through ADR-015 newly added). The remaining item (whether to train with multiple seeds) has been decided by the user in favor of a single seed (ADR-016).
- Claude Code's disposition (ACCEPT/REJECT/DEFER/USER DECISION REQUIRED determinations and rationale) is recorded in `reviews/phase_0_adversarial.md` §11.
- **All design-level blockers have been resolved. Only explicit user approval to transition into Phase 1 remains.**
- **No code implementation has started at all** (`src/`, `configs/`, `notebooks/` do not exist).

## Phase List and Progress

| Phase | Content | Status |
|---|---|---|
| 0 | Design specification drafting | In progress (review reflected, awaiting user approval) |
| 1 | Environment setup + data preparation | Not started |
| 2 | Baseline evaluation | Not started |
| 3 | QLoRA smoke test | Not started |
| 4 | Full QLoRA training | Not started |
| 5 | Evaluation (Base vs Fine-tuned) | Not started |
| 6 | Report / retrospective | Not started |

## Next Actions

1. Present the Codex review disposition (`reviews/phase_0_adversarial.md` §11) and the revised docs to the user, and obtain approval to start Phase 1.
2. Once approval is obtained, start Phase 1 (environment setup + data preparation).

## Unresolved Open Items (Blocking / Open)

See [[EXPERIMENT_SPEC]] §10. In particular, the following need to be resolved around the start of Phase 1:
- Fully-qualified LoRA target module names, target tower, and label masking ([[DECISIONS]] ADR-012, Phase 1 approval gate)
- Finalized prompt template, shot design, and demo-selection procedure ([[DECISIONS]] ADR-009)
- Image resolution and token budget
- Primary metric, improvement threshold X, and paired-bootstrap configuration ([[EVALUATION_PROTOCOL]] §6, §5.1, [[DECISIONS]] ADR-009, ADR-010)
- Production-shape VRAM go/no-go gate measurement ([[DECISIONS]] ADR-014)
- Train/test duplication audit results ([[DECISIONS]] ADR-008)
- Pinning of model/dataset/processor revisions (commit SHA) ([[DECISIONS]] ADR-015)
- The Colab tier that will actually be used

## Change Log

| Date | Change |
|---|---|
| 2026-08-10 | Phase 0 started. Created seven new design docs. Confirmed the existence of the model/dataset. |
| 2026-08-10 | Conducted a Codex adversarial review (`reviews/phase_0_adversarial.md`). Of 12 findings, 11 were ACCEPTed; added ADR-008 through ADR-015; revised [[EXPERIMENT_SPEC]] / [[EVALUATION_PROTOCOL]] / [[IMPLEMENTATION_PLAN]]. The remaining item (multiple seeds) was held as USER DECISION REQUIRED. |
| 2026-08-10 | The user decided to adopt single-seed training (ADR-016). All 12 findings have now been addressed, and the design-level blockers for Phase 0 are resolved. |
| 2026-08-10 | Translated all docs/ files into English at the user's request. |
