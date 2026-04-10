"""Verify that JAX can see a GPU and run computation on it.

Run from the project root:

    uv run python scripts/check_gpu.py

Optionally pin to a specific GPU at runtime:

    CUDA_VISIBLE_DEVICES=3 uv run python scripts/check_gpu.py

This script sets ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` so it plays nicely
on a shared GPU node where other users may be holding memory. It does not
pin a GPU itself — pick one with ``CUDA_VISIBLE_DEVICES`` if needed.
"""

from __future__ import annotations

import os
import sys

# Must be set before importing jax so XLA picks it up.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def main() -> int:
    print(f"JAX version:      {jax.__version__}")
    print(f"JAX backend:      {jax.default_backend()}")
    print(f"JAX devices:      {jax.devices()}")
    print(f"Local device count: {jax.local_device_count()}")

    gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpu_devices:
        print("\nNo GPU visible to JAX. Check that jax[cuda12] is installed")
        print("and that a GPU is available (nvidia-smi).")
        return 1

    print(f"\nRunning a small matmul on {gpu_devices[0]}...")
    key = jax.random.key(0)
    x = jax.random.normal(key, (2048, 2048))
    y = jnp.dot(x, x.T).block_until_ready()
    print(f"  output shape: {y.shape}")
    print(f"  output mean:  {float(y.mean()):.4f}")
    print(f"  device:       {y.device}")

    if y.device.platform != "gpu":
        print("\nComputation did not run on GPU.")
        return 1

    print("\nOK — JAX GPU is working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
