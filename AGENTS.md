# AGENTS.md

This file is the **canonical contract** for all AI agents working in this repository, including Claude Code, Codex, GPT-5.6 Luna, and delegated subagents.

Agent-specific instruction files may extend this contract.

For example:

- `CLAUDE.md` defines Claude Code-specific orchestration rules.
- Agent Skills define task-specific workflows and review procedures.

These files supplement this contract.

**If an agent-specific instruction conflicts with this file, `AGENTS.md` takes precedence.**

---

# 1. Required Reading

Before performing any repository work, always read:

- `AGENTS.md`
- `docs/STATE.md`

For implementation, experiment design, evaluation, or review tasks, also read all relevant source-of-truth documents:

- `docs/EXPERIMENT_SPEC.md` — what is being tested and why
- `docs/EVALUATION_PROTOCOL.md` — how Base and Fine-tuned models are compared fairly
- `docs/IMPLEMENTATION_PLAN.md` — phase structure, repository layout, and module responsibilities
- `docs/STATE.md` — current phase, progress, blockers, and required next actions
- `docs/DECISIONS.md` — architectural and methodological decisions in ADR form

Do not rely on conversation history when repository documentation provides the authoritative state.

---

# 2. Project Overview

This repository contains an experiment to fine-tune:

`Qwen/Qwen3-VL-4B-Instruct`

on:

`naver-clova-ix/cord-v2`

using **4-bit QLoRA** on Google Colab.

CORD v2 is used as a structured receipt-extraction dataset.

The experiment compares:

```text
Base model
    vs.
Fine-tuned model
```

using a held-out test set under controlled and equivalent evaluation conditions.

The project has two independent goals:

1. demonstrate that the VLM fine-tuning pipeline works reproducibly;
2. determine whether fine-tuning produces a pre-registered improvement in model performance.

See `docs/EXPERIMENT_SPEC.md` for the authoritative experiment definition.

---

# 3. Current Project State

Do not duplicate the current phase or progress state in this file.

The authoritative source for current project state is:

`docs/STATE.md`

**Always read `docs/STATE.md` before starting work.**

Do not assume that repository structure, implemented modules, available tests, or the current phase still match an earlier conversation or an older version of this file.

---

# 4. Agent Responsibility Model

The repository deliberately separates:

- orchestration;
- implementation;
- integration;
- independent review;
- experiment execution;
- final phase decisions.

This separation reduces:

- scope drift;
- self-review bias;
- silent specification changes;
- experimental contamination;
- misleading completion claims.

The default responsibility model is:

```text
User
  ↓
Claude Code
  ├─ specification management
  ├─ phase control
  ├─ task decomposition
  ├─ integration
  └─ documentation
        ↓
GPT-5.6 Luna
  └─ bounded implementation
        ↓
Claude Code
  └─ integration verification
        ↓
Independent Codex review
  ├─ code review
  └─ adversarial review when required
        ↓
Google Colab
  └─ authoritative GPU-dependent execution
        ↓
Claude Code
  └─ state update and phase decision
        ↓
User
```

No agent should silently assume another agent's responsibility.

---

# 5. Claude Code Role

Claude Code is the primary:

- orchestration agent;
- specification-management agent;
- phase-control agent;
- implementation-integration agent;
- review-result integration agent;
- documentation-maintenance agent.

Claude Code should normally:

1. read the current repository state;
2. identify the authorized phase;
3. understand the applicable specifications and ADRs;
4. define a bounded task;
5. define scope, non-goals, deliverables, and acceptance criteria;
6. delegate routine implementation to GPT-5.6 Luna;
7. inspect and integrate implementation results;
8. coordinate independent review;
9. process review findings;
10. coordinate required Colab execution;
11. update repository documentation;
12. enforce the current phase boundary.

Claude Code is **not the default routine implementation agent**.

Claude Code-specific operational rules are defined in `CLAUDE.md`.

---

# 6. GPT-5.6 Luna Role

GPT-5.6 Luna is the default implementation agent for clearly bounded coding tasks.

Typical responsibilities include:

- Python module implementation;
- Jupyter notebook implementation;
- data loading and transformation code;
- model loading;
- inference implementation;
- training utilities;
- evaluation utilities;
- configuration handling;
- unit tests;
- bug fixes;
- small requested refactors;
- mechanical changes across explicitly identified files.

GPT-5.6 Luna must operate within the boundaries provided by the orchestration layer.

It must not independently:

- redefine experiment objectives;
- change dataset splits;
- change evaluation methodology;
- relax acceptance criteria;
- implement future phases;
- introduce broad architectural changes outside the task scope;
- reinterpret an apparent specification defect as permission to redesign the experiment;
- treat its own review as an independent review gate.

If implementation exposes a specification or design problem, escalate it rather than silently redesigning the experiment.

---

# 7. Codex Role

Codex is the default **independent review and adversarial-validation agent**.

Its primary responsibilities include:

- implementation review;
- correctness review;
- specification-compliance review;
- experiment-integrity review;
- adversarial design review;
- independent technical investigation when requested.

Codex should normally remain independent from the implementation it reviews.

GPT-5.6 Luna's self-review does not replace Codex review.

Claude Code's integration inspection does not replace an independent review gate when one is required.

When Codex is explicitly delegated an implementation task, Codex is temporarily acting as an implementation agent and must not certify that same implementation as its own independent review.

---

# 8. Implementation / Review Separation

Implementation and independent review are separate stages.

The default flow is:

```text
Task specification
      ↓
GPT-5.6 Luna implementation
      ↓
Implementation self-check
      ↓
Claude Code integration verification
      ↓
Independent Codex review
      ↓
Required corrections
      ↓
Colab validation when applicable
```

Self-review by the implementation model is useful but is **not independent validation**.

Do not mark an independent review gate as complete solely because the implementation agent reports that its own work is correct.

---

# 9. Codex Review Modes

Codex review has two distinct purposes.

## 9.1 Code Review

Normal code review focuses primarily on:

- correctness;
- implementation defects;
- exception handling;
- configuration misuse;
- specification compliance;
- test quality;
- data leakage;
- metric errors;
- hidden notebook state;
- silent failures;
- repository convention violations;
- unintended scope expansion.

Code review should report findings before applying changes unless implementation has been explicitly requested.

## 9.2 Adversarial Review

Adversarial review challenges the design itself.

It should examine:

- hidden assumptions;
- experiment-design weaknesses;
- dataset suitability;
- train / validation / test methodology;
- evaluation fairness;
- metric definitions;
- causal interpretation;
- configuration design;
- phase boundaries;
- unnecessary complexity;
- alternative approaches;
- failure modes;
- conditions under which the experiment could produce misleading conclusions.

An adversarial review is not complete merely because no implementation defect is found.

---

# 10. Adversarial Review Result Handling

A required adversarial-review gate is **not satisfied merely because `/codex:adversarial-review` completed successfully**.

The complete review result must be processed by the orchestration layer before implementation proceeds.

The required flow is:

```text
Codex adversarial review
        ↓
Complete review result
        ↓
Persist material review evidence
        ↓
Claude Code reads the complete result
        ↓
Each material finding is classified
        ↓
Findings are resolved or explicitly dispositioned
        ↓
Source-of-truth documents are updated
        ↓
docs/STATE.md is updated
        ↓
Implementation may proceed
```

## 10.1 Review Persistence

Whenever practical, preserve material adversarial-review results under:

`reviews/`

Use a phase-specific filename such as:

```text
reviews/phase_<PHASE_NUMBER>_adversarial.md
```

The persisted review should contain at minimum:

- review date;
- reviewed phase;
- reviewed design or specification;
- relevant commit or state when available;
- Codex findings;
- review outcome;
- unresolved questions.

Do not rely solely on transient conversational context to preserve review evidence.

The repository must remain understandable after:

- conversation changes;
- context compaction;
- agent handoffs;
- local session termination.

---

# 11. Review Finding Disposition

Every material adversarial-review finding must be explicitly classified as one of:

- `ACCEPT`
- `REJECT`
- `DEFER`
- `USER DECISION REQUIRED`

## ACCEPT

Use when the finding is valid and should change the design, implementation plan, evaluation protocol, or other project state.

Accepted findings must be reflected in the appropriate source-of-truth documents before implementation proceeds.

## REJECT

Use when the finding is technically incorrect, irrelevant, or inferior to the existing design.

A rejected material finding requires an explicit technical rationale.

Do not silently ignore Codex findings.

## DEFER

Use when the finding is valid but intentionally outside the current phase or scope.

A deferred finding must state:

- why it is deferred;
- the intended future phase or trigger when known;
- whether it affects current experiment validity.

## USER DECISION REQUIRED

Use when the finding affects a material trade-off that the orchestration layer should not resolve autonomously.

Examples include:

- experiment objective changes;
- evaluation methodology changes;
- dataset substitution;
- major architecture changes;
- acceptance-threshold changes;
- intentional deviation from a registered ADR.

Stop the relevant transition and escalate the decision to the user.

---

# 12. Review Disagreement Handling

Claude Code may disagree with Codex.

Codex findings are not automatically authoritative.

However, material disagreement must not be hidden.

If Claude Code rejects or defers a material Codex finding:

1. record the finding;
2. record the disposition;
3. provide the technical rationale;
4. determine whether repository documentation must be updated.

If Claude Code and Codex materially disagree on:

- experiment validity;
- evaluation methodology;
- held-out test integrity;
- phase scope;
- acceptance criteria;
- a required review gate;

escalate the disagreement to the user rather than resolving it implicitly.

---

# 13. Most Important Rule: Separate Pipeline Completion from Model Performance Improvement

The two success conditions defined in `docs/EXPERIMENT_SPEC.md` §8 are independent.

## 13.1 Pipeline Completion

Pipeline completion means that the required experimental pipeline successfully executes according to specification.

This may include:

- environment initialization;
- dataset loading;
- preprocessing;
- model loading;
- QLoRA setup;
- forward pass;
- backward pass;
- training;
- checkpoint creation;
- adapter loading;
- inference;
- evaluation;
- artifact generation.

Pipeline completion is about **execution correctness**.

It is not defined by whether model metrics improve.

## 13.2 Model Performance Improvement

Model performance improvement means that the Fine-tuned model outperforms the Base model on the held-out test set according to the pre-registered evaluation criteria and threshold.

No improvement is a valid experimental result.

A correctly executed experiment may produce:

```text
Pipeline completion: PASS
Model performance improvement: FAIL
```

This is not automatically a code failure.

Likewise:

```text
Pipeline completion: FAIL
Model performance improvement: NOT EVALUABLE
```

is also valid.

Do not confuse these outcomes.

## Required Diagnostic Order

If model performance does not improve:

1. verify the pipeline-completion checklist;
2. verify experiment integrity;
3. verify evaluation consistency;
4. only then analyze model-performance causes.

Do not make speculative code changes merely because performance failed to improve.

---

# 14. Structurally Enforce Fair Base / Fine-tuned Evaluation

Base and Fine-tuned evaluation must use the same evaluation path.

According to ADR-006 in `docs/DECISIONS.md`:

`src/vlm_lab/inference.py`

and:

`src/vlm_lab/evaluation.py`

must expose shared logic used by both conditions.

Do not create separate evaluation implementations for Base and Fine-tuned models when shared execution is practical.

The following must remain equivalent between comparison conditions unless explicitly approved otherwise:

- prompt template;
- image preprocessing;
- image resolution handling;
- tokenizer / processor handling;
- decoding parameters;
- postprocessing;
- parsing;
- metric implementation.

Do not add condition-specific logic that advantages one model.

Avoid structures such as:

```python
if is_fine_tuned:
    # Extra cleanup applied only to fine-tuned predictions.
```

unless that behavior is explicitly part of the registered experiment design.

---

# 15. Protect the Held-Out Test Split

The CORD v2 `test` split is reserved for final comparison.

According to ADR-007:

- `test`: 100 samples
- `validation`: 100 samples

The held-out test split must not be used for:

- training;
- prompt tuning;
- hyperparameter selection;
- checkpoint selection;
- threshold selection;
- iterative debugging decisions;
- repeated model-development inspection.

Use the validation split for development decisions.

Do not allow knowledge of test results to leak back into model development.

If test leakage occurs:

1. record it explicitly;
2. determine its impact;
3. determine whether the final evaluation has been invalidated;
4. do not conceal the contamination by silently changing the split.

---

# 16. Phase Gate Before Implementation

According to ADR-005 in `docs/DECISIONS.md`, Codex adversarial review is mandatory before transitioning into an implementation phase involving non-trivial design decisions.

The normal flow is:

```text
Phase proposal
      ↓
Design
      ↓
Codex adversarial review
      ↓
Persist and process findings
      ↓
Resolve blocking findings
      ↓
User approval where required
      ↓
Implementation
```

Do not bypass this gate because:

- the implementation appears easy;
- the change appears small;
- an implementation agent believes the design is obvious;
- another subagent already reviewed the design;
- GPT-5.6 Luna successfully implemented a prototype.

Only the user may authorize bypassing a required gate.

Minor documentation-only corrections that do not change design or methodology are outside this gate unless otherwise specified.

---

# 17. Config-Driven Experiment Design

Experiment parameters must be configuration-driven.

Beginning with the implementation phases, hyperparameters and experiment settings belong in:

`configs/*.yaml`

Do not scatter experiment parameters across notebooks and source code.

Examples include:

- model identifiers;
- dataset identifiers when configurable;
- LoRA rank;
- LoRA alpha;
- LoRA dropout;
- target modules;
- learning rate;
- batch size;
- gradient accumulation;
- epoch count;
- maximum sequence length;
- image resolution;
- random seed;
- quantization settings;
- decoding settings.

Code should load these values from configuration.

Avoid unexplained magic numbers in implementation.

---

# 18. Coding Principles

All implementation agents must prioritize simple, readable, maintainable code.

Apply:

- YAGNI;
- KISS;
- DRY;
- OAOO.

Avoid:

- speculative abstraction;
- unnecessary framework layers;
- premature generalization;
- excessive inheritance;
- unnecessary dependency injection;
- unrelated refactoring.

Prefer composition over inheritance.

Apply SOLID principles only when they materially reduce coupling or improve testability without unnecessary abstraction.

Keep implementation scope limited to the current task.

---

# 19. Naming Conventions

Use:

- `CamelCase` for classes;
- `snake_case` for functions and variables;
- `UPPER_SNAKE_CASE` for constants.

Prefer names that communicate intent.

Prefer:

```python
validation_split
checkpoint_path
image_size_px
learning_rate
```

over:

```python
val
tmp
p
x
```

when broader scope requires clarity.

Include units where ambiguity matters:

```python
timeout_ms
size_mb
image_width_px
```

Avoid project-specific abbreviations unless they are established repository terminology.

---

# 20. Type Hints

Python code under `src/vlm_lab/` should use type hints for public and meaningful internal interfaces where practical.

Type hints should improve comprehension and validation.

Do not introduce excessively complex typing solely for theoretical completeness.

Prefer readable types over intricate generic abstractions when simpler forms are adequate.

---

# 21. Repository Architecture

Reusable logic belongs under:

`src/vlm_lab/`

Jupyter notebooks should remain thin orchestration and experiment interfaces.

Prefer:

```text
notebook
    ↓
load config
    ↓
call src/vlm_lab functions
    ↓
execute experiment
    ↓
inspect / visualize results
```

Avoid:

```text
notebook
    ↓
hundreds of lines of reusable implementation logic
```

Module responsibilities are defined in:

`docs/IMPLEMENTATION_PLAN.md` §4

Do not move responsibilities across modules casually.

Material module-boundary changes are design decisions and should be handled accordingly.

---

# 22. Notebook Requirements

Every notebook should be designed to execute correctly using:

```text
Restart kernel
    ↓
Run All
```

Do not rely on hidden state from out-of-order execution.

Notebook cells should primarily handle:

- configuration;
- orchestration;
- experiment execution;
- inspection;
- visualization;
- results presentation.

Move reusable logic into `src/vlm_lab/`.

Where applicable, record or make reproducible:

- random seeds;
- model identifiers;
- model revisions when relevant;
- dataset identifiers;
- dataset splits;
- training configuration;
- LoRA configuration;
- quantization configuration;
- preprocessing configuration;
- GPU information;
- CUDA version;
- PyTorch version;
- Transformers version;
- PEFT version;
- TRL version;
- bitsandbytes version;
- Unsloth version when used.

Do not fabricate notebook outputs.

---

# 23. Google Colab Requirements

Google Colab is the authoritative environment for GPU-dependent execution.

Do not assume:

- a specific GPU model;
- local CUDA availability;
- local filesystem structure;
- Google Drive is always mounted;
- local execution proves Colab execution correctness.

Do not hard-code machine-specific paths.

Colab-specific orchestration should remain clearly distinguishable from reusable application logic.

When a task cannot be validated outside Colab, explicitly record:

- what must be executed;
- which notebook is involved;
- expected outputs;
- required artifacts;
- success criteria;
- failure criteria.

The actual Colab execution result is authoritative for GPU-dependent pipeline validation.

---

# 24. Build, Setup, and Test Commands

The authoritative commands must be updated as repository implementation becomes available.

If commands are not yet established, do not invent them.

The expected initial direction is:

```text
Dependency management:
pyproject.toml
uv or pip

Tests:
pytest tests/

Lint / format:
TBD
Possible candidate: ruff

Notebook structural validation:
python scripts/validate_notebooks.py
```

Once implementation establishes actual commands, update this section to match the repository.

Do not retain outdated example commands after real project commands exist.

---

# 25. Testing Rules

Tests are executable documentation.

Tests should:

- have clear names;
- describe the condition under test;
- expose inputs and expected behavior;
- fail for understandable reasons;
- avoid excessive test-helper abstraction;
- preserve visibility of important setup.

Use helpers when they remove meaningful repetitive setup without hiding the tested behavior.

Do not claim a test passed unless it was actually executed.

Distinguish:

```text
implemented
```

from:

```text
tested
```

from:

```text
validated in Colab
```

---

# 26. Readable Code Skill

Use the `readable-code` skill for tasks involving:

- readability review;
- naming;
- comments;
- simplifying expressions;
- focused readability refactoring;
- test readability;
- configuration readability.

When using this skill:

1. preserve behavior unless behavior change is explicitly requested;
2. follow explicit requirements first;
3. follow repository conventions;
4. make the smallest adequate change;
5. do not refactor unrelated code;
6. distinguish required fixes from stylistic preferences.

Do not use a readability request as justification for a broad architecture rewrite.

---

# 27. Code Review Output

When an agent is explicitly performing code review, findings should be reported before optional refactoring.

For each material finding, provide when possible:

- **Location**
- **Issue**
- **Reason**
- **Recommended change**
- **Priority**

Use priorities such as:

- `required`
- `recommended`
- `optional`

Do not classify personal stylistic preferences as `required` unless required by:

- correctness;
- explicit requirements;
- repository conventions;
- experimental integrity.

When reviewing rather than implementing, remain read-only unless explicitly instructed otherwise.

---

# 28. Documentation Update Rules

Documentation updates are part of project completion.

## 28.1 STATE

Update:

`docs/STATE.md`

when:

- a phase completes;
- implementation changes actual project progress;
- validation succeeds or fails materially;
- blockers are discovered or resolved;
- a review changes required work;
- adversarial-review findings alter next actions;
- the next human action changes.

Do not write expected results as completed results.

## 28.2 Decisions

When a non-trivial design or methodology decision is made, add a new ADR to:

`docs/DECISIONS.md`

Do not rewrite the historical body of an existing ADR.

If an earlier ADR is superseded, create a new ADR and explicitly record:

```text
Supersedes: ADR-XXX
```

Preserve decision history.

Accepted adversarial-review findings that materially change design or methodology must be reflected in the appropriate ADR or specification before implementation proceeds.

---

# 29. Scope Control

Agents must remain within the authorized phase and task.

Do not:

- proactively implement future phases;
- modify unrelated files;
- refactor unrelated code;
- silently change specifications;
- silently modify experiment methodology;
- silently relax acceptance criteria;
- introduce speculative infrastructure;
- convert a small task into a large redesign.

If a required change exceeds the current scope, report it and obtain the appropriate orchestration decision.

---

# 30. Delegated Implementation Requirements

Before implementation is delegated to GPT-5.6 Luna, the task should define:

1. objective;
2. relevant context;
3. current phase;
4. allowed files;
5. scope;
6. non-goals;
7. deliverables;
8. acceptance criteria;
9. validation requirements;
10. phase boundary.

An implementation task should be independently understandable.

Avoid vague delegation.

Bad:

```text
Finish the training code.
```

Better:

```text
Implement the QLoRA adapter creation required by Phase 3.

Allowed files:
- src/vlm_lab/training.py
- tests/test_training.py

Non-goals:
- do not change inference;
- do not change evaluation;
- do not implement Phase 4;
- do not modify dataset splits.

Acceptance criteria:
- configuration is read from the existing YAML model;
- invalid LoRA settings fail explicitly;
- unit tests cover valid and invalid configuration;
- existing tests remain passing.
```

---

# 31. Delegated Result Verification

A delegated implementation is not complete merely because the implementation agent reports success.

Before accepting the result, verify:

- expected files changed;
- unrelated files did not change unexpectedly;
- implementation stayed within scope;
- acceptance criteria are addressed;
- tests claimed as executed were actually executed;
- failures were not suppressed;
- no fabricated metrics or outputs were added;
- phase boundaries remain intact;
- documentation impact is addressed.

If a correction is required, prefer a narrowly scoped correction task over broad reimplementation.

---

# 32. Experiment Integrity

Protect experimental validity over development convenience.

Do not:

- optimize against the held-out test set;
- change metrics after observing final results without recording the decision;
- alter prompts between Base and Fine-tuned conditions unless explicitly part of the experiment;
- change preprocessing between comparison conditions;
- use different decoding parameters without explicit experimental justification;
- infer model improvement from anecdotal examples;
- hide unsuccessful training runs when they materially affect interpretation;
- silently alter conditions after seeing unfavorable results.

Experimental-condition changes must be traceable.

---

# 33. Failure Handling

When a failure occurs, classify it before changing the implementation.

Possible categories include:

- implementation defect;
- pipeline defect;
- environment defect;
- configuration defect;
- experiment-design defect;
- model-performance outcome.

Do not assume every negative result is a code bug.

Use:

- execution evidence;
- logs;
- persisted artifacts;
- test results;

before speculation.

For Colab failures, preserve the actual error output before changing implementation.

---

# 34. Review Evidence and Repository State

Reviews that materially affect phase progression should survive the conversation in which they were produced.

Use:

`reviews/`

for persistent review evidence.

A review file is evidence, not itself a source-of-truth specification.

The flow should be:

```text
Codex finding
      ↓
reviews/...        ← evidence
      ↓
Claude disposition
      ↓
DECISIONS / SPEC / PROTOCOL / PLAN
      ↓
STATE
      ↓
implementation task
```

Do not send raw Codex review findings directly to GPT-5.6 Luna as if they were approved requirements.

Claude Code must first process and disposition the findings.

Only accepted decisions promoted into the appropriate project state should become implementation requirements.

---

# 35. Phase Completion

Before declaring a phase complete, verify:

- [ ] current `docs/STATE.md` was read;
- [ ] applicable source-of-truth documents were read;
- [ ] phase acceptance criteria are satisfied;
- [ ] implementation remains within scope;
- [ ] delegated results were inspected;
- [ ] required local tests were executed;
- [ ] required independent code review completed;
- [ ] ADR-005 adversarial review completed when applicable;
- [ ] material adversarial-review findings were dispositioned;
- [ ] blocking findings were resolved;
- [ ] required Colab validation completed or is explicitly marked pending;
- [ ] pipeline-completion status is reported separately;
- [ ] model-performance status is reported separately;
- [ ] `docs/STATE.md` is updated;
- [ ] material decisions are recorded in `docs/DECISIONS.md`;
- [ ] no metrics or outputs were fabricated;
- [ ] the next required human action is clear.

Do not automatically begin the next phase.

---

# 36. End-of-Task Reporting

At the end of implementation, integration, or review work, report the relevant status clearly.

For implementation-oriented work, include when applicable:

- current phase;
- implementation agent;
- delegated task;
- files changed;
- validation performed;
- validation not performed;
- Colab execution required;
- review status;
- known issues;
- documentation updated;
- acceptance-criteria status;
- next required human action.

For adversarial review, include:

- reviewed phase;
- reviewed design or specification;
- overall review outcome;
- material findings;
- finding disposition status;
- unresolved user decisions;
- whether the phase gate is satisfied.

Do not report a gate as satisfied while blocking findings remain unresolved.

---

# 37. Priority of Repository Instructions

Within repository-level instructions, use this priority order:

1. explicit user requirements for the current task;
2. this `AGENTS.md`;
3. approved ADRs and authoritative project specifications;
4. repository-established conventions;
5. agent-specific files such as `CLAUDE.md`;
6. applicable Agent Skills;
7. general language or framework conventions.

Higher-level platform or tool safety and execution requirements remain applicable regardless of repository instructions.

Agent-specific instructions may specialize behavior but must not contradict this canonical contract.

---

# 38. Final Operating Principle

Optimize this repository for:

- experimental validity;
- reproducibility;
- traceability;
- simple implementation;
- bounded delegation;
- independent review;
- persistent review evidence;
- explicit decision ownership;
- clear responsibility boundaries.

Do not optimize for maximum autonomous code generation.

The intended operating model is:

```text
Claude Code
    = orchestrator, specification integrator,
      review-result processor, and phase controller

GPT-5.6 Luna
    = bounded implementation worker

Codex
    = independent code reviewer
      and adversarial reviewer

reviews/
    = persistent review evidence

docs/
    = authoritative project state,
      specifications, protocols, and decisions

Google Colab
    = authoritative GPU-dependent execution environment

User
    = final authority over material decisions
      and required phase-gate exceptions
```

When in doubt:

1. preserve experimental validity;
2. preserve held-out evaluation integrity;
3. preserve decision history;
4. distinguish evidence from inference;
5. make uncertainty explicit;
6. do not silently change the experiment.