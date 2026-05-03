"""Modal image for Paper 2: vLLM 0.13.0 + Memento overlay + v3 fork patches.

The image pins:
  - CUDA 12.8 (matches the Memento overlay's expected runtime)
  - vLLM 0.13.0 (the Memento overlay's base version)
  - microsoft/memento at d8c10e6 (the commit our v3 patches are authored against)
  - cainamisir/adaptivecache @ paper2-memento-recall (where our patches live)

Build steps (executed once at image-build time, then cached):
  1. apt deps + python deps
  2. clone our adaptivecache repo at the paper2 branch
  3. clone microsoft/memento at the patch's base commit
  4. apply v3_overlay_patches/ (turns the vanilla overlay into our Phase 1 fork)
  5. run install_overlay.sh (rsyncs Python files into vLLM's site-packages)

After build, the container has:
  - /opt/adaptivecache  — paper2 source tree (paper2-memento-recall branch)
  - /opt/adaptivecache/external/memento — overlay clone with v3 patches applied
  - vLLM in site-packages with our overlay rsynced over it
"""
from __future__ import annotations

import modal


REPO_URL = "https://github.com/cainamisir/adaptivecache.git"
REPO_BRANCH = "paper2-memento-recall"
MEMENTO_COMMIT = "d8c10e6"

# The image builds on a CUDA 12.8 base so the overlay's FlashInfer requirement
# (per HANDOFF.md note 1) works on Hopper/Ampere as well as Blackwell. Modal's
# default Hopper GPUs (H100) provide CUDA via their runtime; the cuda image
# gives us nvcc + headers for any pip-time builds.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "rsync", "curl", "unzip", "build-essential")
    .pip_install(
        "vllm==0.13.0",
        "transformers>=4.40,<5.0",
        "huggingface-hub>=0.20",
        "sentence-transformers>=2.2",
        "anthropic>=0.40",
        "openai>=1.0",
        "numpy",
        "datasets",
        "uv",
    )
    .run_commands(
        # Clone the project repo (paper2 branch — has overlay patches + sources)
        f"cd /opt && git clone --depth 1 -b {REPO_BRANCH} {REPO_URL} adaptivecache",
        # Clone microsoft/memento at the commit our patches are authored against
        f"cd /opt/adaptivecache && mkdir -p external && "
        f"git clone https://github.com/microsoft/memento.git external/memento && "
        f"cd external/memento && git checkout {MEMENTO_COMMIT}",
        # Apply our v3 Phase 1 overlay patches on top of microsoft/memento
        "cd /opt/adaptivecache && bash studies/lifetime_cost/paper2/v3_overlay_patches/apply.sh",
        # Run the overlay installer — rsyncs .py files over vLLM in site-packages
        "cd /opt/adaptivecache/external/memento/vllm && bash install_overlay.sh",
    )
    .env({
        "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        "VLLM_LOGGING_LEVEL": "WARNING",
        "HF_HOME": "/scratch/hf",
        "TRANSFORMERS_CACHE": "/scratch/hf",
        "PYTHONPATH": "/opt/adaptivecache",
    })
)


# Volumes — persist across function invocations.
# Model weights are huge (~60GB for Qwen3-30B-A3B); the swebench repo cache
# avoids re-cloning on every run.
hf_volume = modal.Volume.from_name("paper2-hf-cache", create_if_missing=True)
swebench_volume = modal.Volume.from_name("paper2-swebench-cache", create_if_missing=True)
out_volume = modal.Volume.from_name("paper2-out-v3", create_if_missing=True)


def standard_volumes() -> dict:
    """Return the volume mount dict every paper2 entrypoint should use."""
    return {
        "/scratch/hf": hf_volume,
        "/scratch/swebench_repos": swebench_volume,
        "/scratch/out": out_volume,
    }
