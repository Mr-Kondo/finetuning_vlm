#!/usr/bin/env python
"""Arm C, process 1 (§8.2): build the model, run two optimizer steps, then
save a full trainer-equivalent checkpoint (adapter + optimizer + scheduler +
grad scaler + RNG state + a dummy dataloader position) and exit.

Run as a genuine separate OS process from `notebooks/01b_vram_gate.ipynb`
(via `subprocess.run([sys.executable, "scripts/_vram_gate_c1_save.py", ...])`)
-- a resume that never left the process proves nothing about reload, which is
why arm C is the one exception to "every arm gets its own fresh process"
(docs/proposals/phase1_closure_prereg.md §8.2).

Prints exactly one JSON line to stdout on success, so the orchestrating
notebook cell can parse it without scraping human-readable log text. All
other output goes to stderr-visible prints above that line.

Only synthetic (random) tensors are used -- no CORD v2 data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bitsandbytes as bnb
import torch
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

    with vgc.measured_arm("C1-save") as result:
        model, build_info = vgc.build_model(derived_budget)
        optimizer = bnb.optim.PagedAdamW8bit(
            model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        scaler = GradScaler("cuda")

        # `build_model()` loads the quantized model directly onto the GPU via
        # `device_map` (a bnb 4-bit model cannot be moved post-hoc) -- only the
        # synthetic batch tensors need an explicit `.to("cuda")` here.
        torch.manual_seed(42)
        batch = vgc.build_synthetic_training_batch(build_info, derived_budget)
        batch = {k: v.to("cuda") for k, v in batch.items()}

        for step in range(2):
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss
            if not torch.isfinite(loss):
                print(f"C1 step {step}: non-finite loss {loss.item()}", file=sys.stderr)
                return 1
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(checkpoint_dir / "adapter"))
        torch.save(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(),
                # A dummy dataloader-position integer, per §8.2's checkpoint contents list.
                "dataloader_position": 2,
            },
            checkpoint_dir / "trainer_state.pt",
        )

    measurement = result["measurement"]
    print(json.dumps({"process": "C1-save", **measurement.to_dict()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
