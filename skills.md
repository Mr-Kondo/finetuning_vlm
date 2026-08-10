---
name: readable-code
description: >-
  Review and improve source-code readability through precise naming, focused
  comments, limited scope, readable tests, meaningful file names, and
  maintainable configuration. Use when asked to review code for readability,
  improve naming or comments, simplify complex expressions, or perform a small
  readability-focused refactor. Do not use for unrelated architecture or
  formatting changes.
---

# Readable Code Review and Improvement

## Purpose

Improve readability with the smallest change that resolves the identified
issue. Preserve behavior and follow the target repository's established
conventions.

## Priority

When instructions conflict, apply them in this order:

1. Explicit user requirements.
2. The target repository's existing conventions.
3. The target language and framework conventions.
4. This skill's guidelines.

## Scope

- Do not rename, reformat, or refactor unrelated code.
- Keep changes within the requested scope.
- Preserve behavior unless the user explicitly requests a behavior change.
- Prefer consistency with surrounding code over personal style.
- Distinguish required corrections from optional preferences.

## Workflow

1. Identify the specific readability issue and its location.
2. Check explicit requirements and existing repository conventions.
3. Determine whether a change is necessary or merely stylistic.
4. Propose or apply the smallest adequate change.
5. Check that the change preserves behavior and local consistency.
6. Explain material trade-offs when multiple options are reasonable.

## Review Output

For each finding, provide:

- **Location**: File, symbol, or line range when available.
- **Issue**: The concrete readability problem.
- **Reason**: Why it increases ambiguity or maintenance cost.
- **Recommended change**: The smallest adequate correction.
- **Priority**: `required`, `recommended`, or `optional`.

Do not classify personal preference as `required` unless correctness, an
explicit requirement, or an established repository convention requires it.

## Naming

### Variables

- Use specific names that reveal purpose; avoid generic names such as `tmp` or
  `retval` when a clearer name is practical.
- Include important attributes when they prevent ambiguity:
  - units, such as `timeout_ms` or `size_mb`;
  - state or trust, such as `untrusted_url` or `validated_url`;
  - representation, such as `hex_id`.
- Match name length to scope. Short names such as `i` are acceptable in a small,
  conventional loop; broader scopes need more descriptive names.
- Avoid project-specific or ambiguous abbreviations.
- Prefer values that are assigned once when practical.
- Use an intermediate variable when its name explains intent or decomposes a
  complex expression. Remove it when it only repeats the expression.

### Functions

- Use verbs that describe the operation precisely. Prefer `fetch`, `download`,
  `record`, or `increment` when `get` or `count` would be ambiguous.
- Respect established semantic expectations. For example, do not name an
  expensive remote operation as a lightweight accessor when the repository
  treats `get*` functions as inexpensive.
- For boolean results, use names whose `true` value is clear, commonly with
  `is`, `has`, `can`, or `should`, when consistent with the language.
- Prefer affirmative names when they avoid double negatives.
- Remove words that add no meaning.

### Types, Classes, Modules, and Packages

- Name components after their concrete responsibility or domain concept.
- Avoid vague names such as `Core`, `Processor`, `Manager`, or `Utils` unless
  the surrounding domain gives them a precise, established meaning.
- Follow the language and repository's casing and member-variable conventions.
  Do not introduce a new naming scheme into an existing codebase.

### Files

- Name a file after its primary responsibility.
- Avoid broad names such as `utils`, `common`, or `misc` when a more specific
  name is available.
- Follow the repository's casing, separators, suffixes, and extension patterns.
- Use the same pattern for files of the same category, such as tests or
  configuration files.

## Structure and Expressions

- Optimize for the time another developer needs to understand and safely change
  the code.
- Keep variable scope and lifetime as small as practical.
- Move definitions close to their use unless doing so harms structure.
- Break complex expressions into named parts when this exposes intent.
- Keep each function or logical block focused on one task.
- Group related statements and separate distinct steps with blank lines.
- Extract a helper only when it removes meaningful duplication, names a useful
  concept, or isolates a distinct responsibility.
- Avoid speculative abstraction and unnecessary implementation.
- Use column alignment only when it improves scanning and the formatter will
  preserve it; do not create noisy diffs for manual alignment.

## Comments

- Explain **why**, constraints, trade-offs, surprising behavior, or historical
  context that the code cannot express directly.
- Do not restate obvious code behavior.
- Explain the source or constraint behind non-obvious constants.
- Warn about important performance characteristics or misuse risks.
- Use precise nouns instead of ambiguous pronouns such as “it” or “this.”
- Keep comments concise and update or remove them when the code changes.
- Use `TODO`, `FIXME`, `HACK`, or `XXX` only according to repository convention;
  include an issue reference or actionable condition when practical.
- In languages without named arguments, parameter comments may clarify
  otherwise ambiguous literals, for example:
  `Connect(/* timeout_ms = */ 10, /* use_encryption = */ false);`

## Tests

- Treat tests as executable documentation.
- Name tests according to repository convention so the unit, condition, and
  expected behavior are clear.
- Keep setup focused and make inputs and expected outputs visible.
- Use helpers to remove repetitive setup without hiding the behavior under test.
- Prefer focused tests that fail for one understandable reason.
- Provide failure messages that help diagnose the mismatch.

## Configuration

Apply these guidelines when configuration is part of the requested review:

- Treat configuration as version-controlled, reviewable, and testable code.
- Group configuration by the domain concept that changes together.
- Validate configuration with an appropriate schema or parser when practical.
- For named JSON entities, an object keyed by name can improve direct access
  when order and duplicate names are irrelevant. Use an array when ordering or
  duplication matters.
- Correct naming or structural inconsistency in the underlying model rather
  than adding compensating complexity to its representation.
- Version breaking changes to shared configuration libraries or schemas.

## Final Check

Before completing the task, verify that:

- the requested readability issue is resolved;
- behavior is preserved;
- the change follows repository conventions;
- no unrelated code was modified;
- comments explain information not already evident from the code;
- the result is simpler to understand, not merely different.