"""Shared plumbing for the Phase 1 closure VRAM go/no-go gate (§8.2).

Used by `notebooks/01b_vram_gate.ipynb` and by the two arm-C subprocess
scripts (`_vram_gate_c1_save.py`, `_vram_gate_c2_resume.py`). This module is
deliberately narrow: it exists only because arm C genuinely needs to run in
two separate OS processes (a resume that never left the process proves
nothing about reload -- see `docs/proposals/phase1_closure_prereg.md` §8.2),
which rules out defining this logic only inside notebook cells. It is NOT a
general checkpoint-resume framework, and it does not belong in
`src/vlm_lab/` -- this task's scope explicitly excludes touching
`src/vlm_lab/*.py`, and this scaffolding is specific to one Phase 1 closure
gate, not reusable experiment logic.

Every tensor built here is synthetic (random). This module never loads CORD
v2 data, images, or ground truth -- see AGENTS.md §15 / §32 and the gate's
non-goals.
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import threading
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Cross-checked against docs/proposals/phase1_closure_prereg.md §1 (ADR-015). If
# configs/derived_budget.yaml's model_revision ever disagrees with this, that is
# a real drift between notebook 01a's output and the pre-registered pin -- fail
# loudly rather than silently trusting whichever value happened to be on disk.
EXPECTED_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

GIB = 1024**3
# §8.2 step 5: a fixed, pre-registered 1.0 GiB safety margin -- not a residual
# inferred after the fact -- applied to BOTH the allocator inequality and the
# whole-device free-memory inequality.
SAFETY_MARGIN_BYTES = 1.0 * GIB
FREE_MEMORY_SAMPLE_INTERVAL_S = 0.1  # 10 Hz, per §8.2 step 5


# --------------------------------------------------------------------------
# configs/derived_budget.yaml contract loader
# --------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "generated_at_utc",
    "model_revision",
    "dataset_revision",
    "processor_size",
    "patch_size",
    "spatial_merge_size",
    "image_token_ceiling",
    "fixed_prompt_and_template_tokens",
    "eval_prefix_upper_bound",
    "max_new_tokens",
    "max_seq_len",
    "measured_distributions",
    "schema",
    "lora",
)
_REQUIRED_LORA_KEYS = ("target_modules", "r", "alpha", "dropout", "expected_trainable_params")
_REQUIRED_DISTRIBUTION_KEYS = ("eval_prefix_len", "train_seq_len", "assistant_label_n", "image_token_n")

_MISSING_FILE_MESSAGE = (
    "configs/derived_budget.yaml does not exist at {path!r}.\n\n"
    "This file is written by the COMPANION notebook "
    "`notebooks/01a_closure_measurements.ipynb`, which must be run to completion "
    "on Colab (real GPU + real processor measurements over train+validation) "
    "BEFORE this VRAM gate notebook can size arms B and D.\n\n"
    "This notebook will NOT guess a fallback token budget or LoRA configuration "
    "-- doing so would size the gate against invented numbers instead of the "
    "measured production shape, defeating its entire purpose "
    "(docs/proposals/phase1_closure_prereg.md §8.2, §9 deliverable 8).\n\n"
    "Run notebooks/01a_closure_measurements.ipynb first, then re-run this notebook."
)


def load_derived_budget(path: str | Path = "configs/derived_budget.yaml") -> dict[str, Any]:
    """Load and minimally validate configs/derived_budget.yaml's frozen contract.

    Raises a clear, actionable error (naming the companion notebook) if the
    file is missing, and a clear structural error if it exists but does not
    match the schema this gate depends on. Never falls back to hardcoded
    defaults -- a missing or malformed budget means the gate cannot
    legitimately size arms B and D, and must say so rather than guess.
    """
    budget_path = Path(path)
    if not budget_path.is_file():
        raise FileNotFoundError(_MISSING_FILE_MESSAGE.format(path=str(budget_path.resolve())))

    with budget_path.open("r", encoding="utf-8") as f:
        budget = yaml.safe_load(f)

    if not isinstance(budget, dict):
        raise ValueError(
            f"{budget_path} did not parse into a mapping (got {type(budget).__name__}). "
            "It is not the file notebooks/01a_closure_measurements.ipynb is expected to write."
        )

    missing_top = [key for key in _REQUIRED_TOP_LEVEL_KEYS if key not in budget]
    if missing_top:
        raise ValueError(
            f"{budget_path} is missing required top-level key(s) {missing_top}. "
            "The schema is frozen in this gate's delegated task spec and in "
            "docs/proposals/phase1_closure_prereg.md's §8.2/§9 contract -- either "
            "notebook 01a wrote an incomplete file, or the contract has drifted."
        )

    if budget["schema_version"] != 1:
        raise ValueError(
            f"{budget_path} has schema_version={budget['schema_version']!r}, expected 1. "
            "This gate was written against schema_version 1 and refuses to guess "
            "compatibility with a different schema."
        )

    if budget["model_revision"] != EXPECTED_MODEL_REVISION:
        raise ValueError(
            f"{budget_path}'s model_revision={budget['model_revision']!r} does not match "
            f"the pre-registered revision {EXPECTED_MODEL_REVISION!r} "
            "(docs/proposals/phase1_closure_prereg.md §1 / ADR-015). Refusing to proceed "
            "against an unexpected model revision."
        )

    missing_lora = [key for key in _REQUIRED_LORA_KEYS if key not in budget["lora"]]
    if missing_lora:
        raise ValueError(f"{budget_path}'s `lora:` block is missing key(s) {missing_lora}.")

    missing_dist = [key for key in _REQUIRED_DISTRIBUTION_KEYS if key not in budget["measured_distributions"]]
    if missing_dist:
        raise ValueError(f"{budget_path}'s `measured_distributions:` block is missing key(s) {missing_dist}.")

    return budget


# --------------------------------------------------------------------------
# §8.2 measurement protocol
# --------------------------------------------------------------------------


@dataclasses.dataclass
class ArmMeasurement:
    """One arm's §8.2 measurement result."""

    arm_name: str
    baseline_free_bytes: int
    total_bytes: int
    max_memory_allocated_bytes: int
    max_memory_reserved_bytes: int
    min_free_observed_bytes: int
    allocator_margin_pass: bool  # condition (a)
    global_free_margin_pass: bool  # condition (b)

    @property
    def go(self) -> bool:
        return self.allocator_margin_pass and self.global_free_margin_pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_name": self.arm_name,
            "baseline_free_gib": self.baseline_free_bytes / GIB,
            "total_gib": self.total_bytes / GIB,
            "max_memory_allocated_gib": self.max_memory_allocated_bytes / GIB,
            "max_memory_reserved_gib": self.max_memory_reserved_bytes / GIB,
            "min_free_observed_gib": self.min_free_observed_bytes / GIB,
            "allocator_margin_pass": self.allocator_margin_pass,
            "global_free_margin_pass": self.global_free_margin_pass,
            "go": self.go,
        }


class _FreeMemorySampler(threading.Thread):
    """Background thread sampling `torch.cuda.mem_get_info()`'s free bytes at
    10 Hz, tracking the running minimum -- so a transient dip between two
    end-of-arm readings is caught rather than missed (§8.2 step 5)."""

    def __init__(self, interval_s: float = FREE_MEMORY_SAMPLE_INTERVAL_S) -> None:
        super().__init__(daemon=True)
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self.min_free_bytes: int | None = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            if self.min_free_bytes is None or free_bytes < self.min_free_bytes:
                self.min_free_bytes = free_bytes
            self._stop_event.wait(self._interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5.0)


def _print_arm_report(measurement: ArmMeasurement) -> None:
    m = measurement
    print(f"--- {m.arm_name} measurement report ---")
    print(f"  baseline_free:          {m.baseline_free_bytes / GIB:.3f} GiB")
    print(f"  total device memory:    {m.total_bytes / GIB:.3f} GiB")
    print(f"  max_memory_allocated:   {m.max_memory_allocated_bytes / GIB:.3f} GiB")
    print(f"  max_memory_reserved:    {m.max_memory_reserved_bytes / GIB:.3f} GiB")
    print(f"  min_free_observed:      {m.min_free_observed_bytes / GIB:.3f} GiB")
    print(
        f"  (a) max_reserved <= baseline_free - 1.0 GiB: "
        f"{'PASS' if m.allocator_margin_pass else 'FAIL'} "
        f"({m.max_memory_reserved_bytes / GIB:.3f} <= {(m.baseline_free_bytes - SAFETY_MARGIN_BYTES) / GIB:.3f})"
    )
    print(
        f"  (b) min_free_observed >= 1.0 GiB:            "
        f"{'PASS' if m.global_free_margin_pass else 'FAIL'} "
        f"({m.min_free_observed_bytes / GIB:.3f} >= {SAFETY_MARGIN_BYTES / GIB:.3f})"
    )
    print(f"  => {m.arm_name} GO/NO-GO: {'GO' if m.go else 'NO-GO'}")


@contextlib.contextmanager
def measured_arm(arm_name: str):
    """§8.2 steps 2-5 measurement protocol for one VRAM gate arm.

    Resets peak allocator stats, samples the global free-memory low-water
    mark at 10 Hz for the duration of the block, and on exit reports and
    returns an `ArmMeasurement` via the yielded dict's "measurement" key.

    Deviation from the frozen protocol, stated plainly rather than silently
    claimed as met: §8.2 step 1 calls for each arm to run in a fresh OS
    process. Arms A/B/D below run inside one shared kernel instead (with
    explicit `del` + `gc.collect()` + `torch.cuda.empty_cache()` +
    `torch.cuda.reset_peak_memory_stats()` between them), which is an
    approximation -- CUDA context overhead and allocator fragmentation from
    an earlier arm are not fully guaranteed to be released. Arm C is the one
    exception: it genuinely launches two separate subprocesses, because a
    resume that never left the process proves nothing about reload.
    """
    baseline_free_bytes, total_bytes = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    sampler = _FreeMemorySampler()
    sampler.start()
    result: dict[str, Any] = {}
    try:
        yield result
    finally:
        torch.cuda.synchronize()
        sampler.stop()
        max_allocated = torch.cuda.max_memory_allocated()
        max_reserved = torch.cuda.max_memory_reserved()
        min_free_observed = sampler.min_free_bytes if sampler.min_free_bytes is not None else baseline_free_bytes

        measurement = ArmMeasurement(
            arm_name=arm_name,
            baseline_free_bytes=baseline_free_bytes,
            total_bytes=total_bytes,
            max_memory_allocated_bytes=max_allocated,
            max_memory_reserved_bytes=max_reserved,
            min_free_observed_bytes=min_free_observed,
            allocator_margin_pass=max_reserved <= baseline_free_bytes - SAFETY_MARGIN_BYTES,
            global_free_margin_pass=min_free_observed >= SAFETY_MARGIN_BYTES,
        )
        result["measurement"] = measurement
        _print_arm_report(measurement)


# --------------------------------------------------------------------------
# Model construction (§2, §3, §7.3 frozen order) -- arms A, B, C, D all use this
# --------------------------------------------------------------------------


def build_base_model(derived_budget: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Steps 1-4 of the frozen §7.3 order: load the 4-bit NF4 quantized base
    model and run `prepare_model_for_kbit_training`. No adapter yet.

    Split out from `build_model()` so arm C's resume process (C2) can load
    the SAME base model and then attach the SAVED adapter via
    `PeftModel.from_pretrained`, instead of injecting a second, different
    fresh adapter on top of one already on disk.

    Returns (model, build_info) -- `build_info` at this stage carries the
    dtype/SDPA/GPU/vision-geometry facts; the ADR-012 trainable-parameter
    evidence is added once an adapter exists (see `inject_fresh_adapter`).
    """
    model_revision = derived_budget["model_revision"]

    config = AutoConfig.from_pretrained(MODEL_ID, revision=model_revision)
    text_config = config.get_text_config()
    real_max_position_embeddings = getattr(text_config, "max_position_embeddings", None)
    if real_max_position_embeddings is None:
        raise RuntimeError(
            f"Could not introspect max_position_embeddings from {type(text_config).__name__} "
            "-- the §4 assertion cannot run without it. This is a real finding: either the "
            "pinned transformers version changed this config's field name, or the model class "
            "changed. Do not guess a value."
        )
    max_seq_len = derived_budget["max_seq_len"]
    if max_seq_len > real_max_position_embeddings:
        raise RuntimeError(
            f"configs/derived_budget.yaml's max_seq_len={max_seq_len} exceeds the pinned "
            f"model's max_position_embeddings={real_max_position_embeddings}. §4's assertion "
            "failed -- do not proceed."
        )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # bitsandbytes 4-bit quantization happens at load time and must target the
    # GPU directly -- a CPU-loaded bnb 4-bit model cannot be moved with a
    # later `.to("cuda")` call (unsupported/undefined for bnb's Params4bit).
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=model_revision,
        quantization_config=bnb_config,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": torch.cuda.current_device()},
    )

    dtype_counts: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1

    sdpa_backend = _introspect_sdpa_backend()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_capability = torch.cuda.get_device_capability(0)

    print(f"[build_base_model] realized module dtypes (dtype -> module-parameter-tensor count): {dtype_counts}")
    print(f"[build_base_model] attn_implementation requested: sdpa; enabled SDPA backends: {sdpa_backend}")
    print(f"[build_base_model] GPU: {gpu_name}, compute capability: {gpu_capability}")

    # Frozen order (§7.3): k-bit prep BEFORE adapter injection.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    build_info = {
        "dtype_counts": dtype_counts,
        "sdpa_backend": sdpa_backend,
        "gpu_name": gpu_name,
        "gpu_capability": list(gpu_capability),
        "vision_patch_size": config.vision_config.patch_size,
        "vision_temporal_patch_size": config.vision_config.temporal_patch_size,
        "vision_in_channels": config.vision_config.in_channels,
        "vision_spatial_merge_size": config.vision_config.spatial_merge_size,
        "image_token_id": config.image_token_id,
        "video_token_id": config.video_token_id,
        "vocab_size": text_config.vocab_size,
    }
    return model, build_info


def inject_fresh_adapter(
    base_model: Any, build_info: dict[str, Any], derived_budget: dict[str, Any], adapter_seed: int = 42
) -> Any:
    """Steps 5-8 of the frozen §7.3 order: inject a freshly-initialized LoRA
    adapter (§3.3), set `use_cache = False`, and run the ADR-012
    approval-gate assertions against the real constructed model.

    Mutates `build_info` in place to add the ADR-012 evidence
    (`trainable_params`, `trainable_param_prefix`). Raises `AssertionError`
    if either ADR-012 condition fails -- these are the approval-gate
    assertions and must not be silently downgraded to a warning.
    """
    lora_cfg = derived_budget["lora"]
    torch.manual_seed(adapter_seed)
    lora_config = LoraConfig(
        target_modules=lora_cfg["target_modules"],
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        use_rslora=False,
        init_lora_weights=True,
    )
    model = get_peft_model(base_model, lora_config)
    model.config.use_cache = False

    # --- ADR-012 approval-gate assertions: must actually run against the real model. ---
    trainable_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for _, p in trainable_params)
    expected_count = lora_cfg["expected_trainable_params"]
    if trainable_count != expected_count:
        print(
            f"[ADR-012 FINDING] trainable parameter count mismatch: got {trainable_count}, "
            f"expected {expected_count} from configs/derived_budget.yaml."
        )
        raise AssertionError(
            f"ADR-012 approval-gate FAILED: trainable parameter count is {trainable_count}, "
            f"expected {expected_count}. Do not silently accept this -- the LoRA arithmetic "
            "or the target_modules regex may not be matching what "
            "docs/proposals/phase1_closure_prereg.md §3.3 assumed."
        )

    # NOTE: get_peft_model() wraps the base model, so the REAL parameter-name
    # prefix is "base_model.model." + the original module path, not the bare
    # "model.language_model." the target_modules regex was written against.
    # Assert on module identity (contains ".language_model.", never
    # ".visual.") rather than a hardcoded prefix string, and record the
    # actual observed prefix for the record.
    outside_language_tower = [
        name for name, _ in trainable_params if ".language_model." not in name or ".visual." in name
    ]
    if outside_language_tower:
        print(f"[ADR-012 FINDING] trainable parameters outside the language tower: {outside_language_tower}")
        raise AssertionError(
            "ADR-012 approval-gate FAILED: found trainable parameters outside the language "
            f"tower: {outside_language_tower}"
        )
    observed_prefix = trainable_params[0][0].rsplit(".language_model.", 1)[0] + ".language_model."
    print(
        f"[inject_fresh_adapter] ADR-012 PASS: {trainable_count} trainable params (expected "
        f"{expected_count}), all under the real observed prefix {observed_prefix!r}."
    )

    build_info["trainable_params"] = trainable_count
    build_info["trainable_param_prefix"] = observed_prefix
    return model


def build_model(derived_budget: dict[str, Any], adapter_seed: int = 42) -> tuple[Any, dict[str, Any]]:
    """Build the 4-bit NF4 quantized base model with a freshly-initialized
    LoRA adapter, following the frozen §7.3 order in full (`build_base_model`
    then `inject_fresh_adapter`). Used by arms A, B, and D, which each need a
    complete model with a fresh adapter. Arm C's resume process (C2) instead
    calls `build_base_model` directly and attaches a SAVED adapter.
    """
    model, build_info = build_base_model(derived_budget)
    model = inject_fresh_adapter(model, build_info, derived_budget, adapter_seed=adapter_seed)
    return model, build_info


def _introspect_sdpa_backend() -> dict[str, bool]:
    """Best-effort report of which SDPA backends are ENABLED on this device.

    This is not a guarantee of which backend PyTorch actually dispatched for
    any specific call -- that requires profiling each call (e.g.
    `torch.profiler`) and inspecting the emitted kernel names, which this
    gate does not do. Reported here as the honest limit of what a cheap,
    non-profiling introspection can say (§2 P-2b requires recording the
    realized backend; this is the best-effort version of that when a full
    profiler trace is out of scope for a memory gate).
    """
    return {
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
    }


# --------------------------------------------------------------------------
# Synthetic batch construction (production shape, dummy content only)
# --------------------------------------------------------------------------


def _factor_merged_grid(n_merged_tokens: int) -> tuple[int, int]:
    """Factor n_merged_tokens into (h_m, w_m) as close to square as possible,
    for building a single-image grid_thw = [1, h_m*merge, w_m*merge]."""
    h_m = int(math.isqrt(n_merged_tokens))
    while h_m > 1 and n_merged_tokens % h_m != 0:
        h_m -= 1
    w_m = n_merged_tokens // h_m
    return h_m, w_m


def build_synthetic_image(build_info: dict[str, Any], n_merged_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build random pixel_values/image_grid_thw for one synthetic image whose
    merged (post spatial-merge) visual token count is exactly n_merged_tokens.

    Shapes mirror the REAL processor's output format for
    Qwen3VLVisionPatchEmbed: pixel_values is (num_patches, in_channels *
    temporal_patch_size * patch_size**2), grid_thw is (1, 3) = [t, h, w] in
    pre-merge patch units.
    """
    merge = build_info["vision_spatial_merge_size"]
    h_m, w_m = _factor_merged_grid(n_merged_tokens)
    grid_thw = torch.tensor([[1, h_m * merge, w_m * merge]], dtype=torch.long)
    num_patches = int(grid_thw.prod(dim=-1).item())
    patch_dim = (
        build_info["vision_in_channels"] * build_info["vision_temporal_patch_size"] * build_info["vision_patch_size"] ** 2
    )
    pixel_values = torch.randn(num_patches, patch_dim, dtype=torch.float32)
    assert num_patches // (merge**2) == n_merged_tokens
    return pixel_values, grid_thw


def build_synthetic_training_batch(
    build_info: dict[str, Any],
    derived_budget: dict[str, Any],
    image_token_budget: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build ONE synthetic training example at production shape (§8.2 arm B).

    Dummy/random content only -- no real images, no real CORD ground truth.
    `image_token_budget` overrides `image_token_ceiling` (used for arm B's
    half-budget timing baseline); `input_ids` length is always
    `derived_budget["max_seq_len"]`, which per §4 already counts the expanded
    image placeholder tokens.

    Includes `mm_token_type_ids` (0=text, 1=image), which the real
    Qwen3VLModel.get_rope_index()/compute_3d_position_ids() REQUIRES whenever
    `image_grid_thw` is passed alongside `input_ids` -- confirmed by direct
    inspection of transformers==5.15.0's modeling_qwen3_vl.py and by an
    executed structural test; this field is not listed in this gate's
    delegated task spec's arm-B field list, which appears to be a real gap in
    that spec rather than an optional field.
    """
    vocab_size = build_info["vocab_size"]
    image_token_ceiling = image_token_budget if image_token_budget is not None else derived_budget["image_token_ceiling"]
    max_seq_len = derived_budget["max_seq_len"]
    assistant_label_n_max = derived_budget["measured_distributions"]["assistant_label_n"]["max"]
    image_token_id = build_info["image_token_id"]
    video_token_id = build_info["video_token_id"]

    if image_token_ceiling >= max_seq_len:
        raise ValueError(
            f"image_token_budget={image_token_ceiling} >= max_seq_len={max_seq_len}; "
            "cannot place the image block plus any text inside the sequence."
        )

    pixel_values, grid_thw = build_synthetic_image(build_info, image_token_ceiling)

    input_ids = torch.randint(0, vocab_size, (1, max_seq_len), dtype=torch.long)
    input_ids[input_ids == image_token_id] = 0
    input_ids[input_ids == video_token_id] = 0
    image_start = 1  # position 0 left as a distinct "system start" text token
    input_ids[0, image_start : image_start + image_token_ceiling] = image_token_id

    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[0, image_start : image_start + image_token_ceiling] = 1

    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    labels[:, :-assistant_label_n_max] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
        "labels": labels,
    }


def build_synthetic_eval_prefix(build_info: dict[str, Any], derived_budget: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Build a synthetic evaluation prefix at `eval_prefix_upper_bound` length
    (§8.2 arm D): image tokens = `image_token_ceiling`, text tokens =
    `fixed_prompt_and_template_tokens`. No labels (generation input only)."""
    vocab_size = build_info["vocab_size"]
    image_token_ceiling = derived_budget["image_token_ceiling"]
    fixed_text_tokens = derived_budget["fixed_prompt_and_template_tokens"]
    eval_prefix_upper_bound = derived_budget["eval_prefix_upper_bound"]
    image_token_id = build_info["image_token_id"]
    video_token_id = build_info["video_token_id"]

    expected_len = image_token_ceiling + fixed_text_tokens
    if expected_len != eval_prefix_upper_bound:
        print(
            f"[FINDING] image_token_ceiling + fixed_prompt_and_template_tokens = {expected_len} "
            f"!= eval_prefix_upper_bound = {eval_prefix_upper_bound} in configs/derived_budget.yaml. "
            "Using eval_prefix_upper_bound as the sequence length per §4's definition; the "
            "mismatch itself is reported here rather than silently reconciled."
        )

    pixel_values, grid_thw = build_synthetic_image(build_info, image_token_ceiling)

    input_ids = torch.randint(0, vocab_size, (1, eval_prefix_upper_bound), dtype=torch.long)
    input_ids[input_ids == image_token_id] = 0
    input_ids[input_ids == video_token_id] = 0
    image_start = 1
    input_ids[0, image_start : image_start + image_token_ceiling] = image_token_id

    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[0, image_start : image_start + image_token_ceiling] = 1

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
    }


def free_cuda_memory() -> None:
    """Between-arms cleanup: reclaim allocator state after the caller has
    already dropped its own references.

    NOTE: this function cannot `del` anything on the caller's behalf --
    `del` inside this function only removes a local binding in this
    function's own frame, not the caller's variable, so callers MUST `del`
    their model/optimizer/tensor variables themselves immediately before
    calling this. (`vgc.free_cuda_memory(model_a)` would silently fail to
    free `model_a` -- an earlier version of this function had exactly that
    bug.) Approximates a fresh process (see `measured_arm`'s deviation
    note); it is not a substitute for genuine process isolation, only the
    closest approximation practical within one shared kernel.
    """
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def median_step_time(step_times_s: list[float], warmup_steps: int = 2) -> float:
    """§8.2 step 6: discard `warmup_steps` warm-up steps, return the median of
    the remainder."""
    timed = step_times_s[warmup_steps:]
    if not timed:
        raise ValueError(f"Need more than {warmup_steps} recorded step times, got {len(step_times_s)}.")
    return statistics_median(timed)


def statistics_median(values: list[float]) -> float:
    import statistics

    return statistics.median(values)


def now_s() -> float:
    return time.time()
