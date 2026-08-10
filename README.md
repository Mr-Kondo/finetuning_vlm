# vlm-lab

An experiment to fine-tune `Qwen/Qwen3-VL-4B-Instruct` on `naver-clova-ix/cord-v2`
using 4-bit QLoRA on Google Colab, comparing the base model against the
fine-tuned model on a held-out test set.

This README is intentionally minimal. The canonical source of truth for the
project contract, specifications, phase status, and decisions lives in
`AGENTS.md` and `docs/` (see in particular `docs/EXPERIMENT_SPEC.md`,
`docs/IMPLEMENTATION_PLAN.md`, and `docs/STATE.md`).

## Install

```bash
pip install -e ".[dev]"
```
