#!/usr/bin/env python
"""Arm C, process 2 (§8.2): start cold, load the checkpoint written by
`_vram_gate_c1_save.py` (adapter + optimizer + scheduler-equivalent state +
grad scaler + RNG state + dataloader position), run ONE further optimizer
step, and report its own peak memory.

Run as a genuine separate OS process from `notebooks/01b_vram_gate.ipynb`,
started cold via `subprocess.run([sys.executable,
"scripts/_vram_gate_c2_resume.py", ...])` -- this process never saw C1's
Python state, only the files C1 wrote to `--checkpoint-dir`.

Prints exactly one JSON line to stdout on success, so the orchestrating
notebook cell can parse it without scraping human-readable log text.

Only synthetic (random) tensors are used -- no CORD v2 data. This script has
no scheduler state to load because the frozen training config
(docs/proposals/phase1_closure_prereg.md §7.3) uses `lr_scheduler_type:
cosine` computed from `TrainingArguments`, which this narrow gate script does
not reconstruct; its absence does not change adapter/optimizer/scaler/RNG
memory footprint, which is what this arm measures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bitsandbytes as bnb
import torch
from peft import PeftModel
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vram_gate_common as vgc  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--derived-budget", default="configs/derived_budget.yaml")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available in this process; arm C cannot run.", file=sys.stderr)
        return 1

    derived_budget = vgc.load_derived_budget(args.derived_budget)
    checkpoint_dir = Path(args.checkpoint_dir)

    with vgc.measured_arm("C2-resume") as result:
        base_model, build_info = vgc.build_base_model(derived_budget)
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir / "adapter"), is_trainable=True)
        model.config.use_cache = False

        optimizer = bnb.optim.PagedAdamW8bit(
            model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        scaler = GradScaler("cuda")

        trainer_state = torch.load(checkpoint_dir / "trainer_state.pt", weights_only=False)
        optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
        scaler.load_state_dict(trainer_state["scaler_state_dict"])
        torch.set_rng_state(trainer_state["torch_rng_state"])
        torch.cuda.set_rng_state(trainer_state["cuda_rng_state"])
        dataloader_position = trainer_state["dataloader_position"]

        torch.manual_seed(42 + dataloader_position)
        batch = vgc.build_synthetic_training_batch(build_info, derived_budget)
        batch = {k: v.to("cuda") for k, v in batch.items()}

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss
        if not torch.isfinite(loss):
            print(f"C2 resumed step: non-finite loss {loss.item()}", file=sys.stderr)
            return 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    measurement = result["measurement"]
    print(json.dumps({"process": "C2-resume", **measurement.to_dict()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
