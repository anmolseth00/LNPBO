"""Compute-device selection and NumPy->Tensor conversion for the GP-BO stack.

Split out of ``gp_bo.py``. torch is an optional dependency loaded lazily by the
parent package, so importing this module already implies torch is available.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch

_DEVICE_LOGGED = False

# Minimum free VRAM (GiB) required before claiming CUDA. Tuned so that 4-8
# parallel ablation workers can coexist on a 24 GiB GPU without one of them
# tripping OOM at model.to(device). Override via LNPBO_GPU_MIN_FREE_GIB.
_DEFAULT_GPU_MIN_FREE_GIB = 2.0


def _gpu_has_headroom(min_free_gib: float) -> tuple[bool, float]:
    """Return (ok, free_gib) where ok=True means CUDA has min_free_gib free.

    Uses ``torch.cuda.mem_get_info()`` which reports the driver's view of
    free VRAM, accounting for other processes (other ablation workers,
    chemprop subprocesses, anything else holding GPU memory).
    """
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
    except (RuntimeError, AssertionError):
        return False, 0.0
    free_gib = free_bytes / (1024**3)
    return free_gib >= min_free_gib, free_gib


def get_device(use_mps: bool | None = None) -> torch.device:
    """Select compute device with memory-aware CUDA fallback.

    Priority:
      1. ``LNPBO_FORCE_CPU=1`` env var → CPU (hard override for parallel runs).
      2. CUDA if available AND has at least ``LNPBO_GPU_MIN_FREE_GIB``
         (default 2 GiB) free - prevents OOM when other workers are running.
      3. CPU - reliable float64, always works.
      4. MPS (opt-in via ``LNPBO_USE_MPS``) - Apple Silicon, float32 only.

    On the first call of a process, prints a one-line confirmation of the
    selected device so users can verify GPU engagement from logs
    without resorting to nvidia-smi or py-spy.
    """
    force_cpu = os.environ.get("LNPBO_FORCE_CPU", "").lower() in {"1", "true", "yes"}

    device: torch.device
    fallback_reason = ""
    if force_cpu:
        device = torch.device("cpu")
        fallback_reason = "LNPBO_FORCE_CPU"
    elif torch.cuda.is_available():
        try:
            min_free = float(os.environ.get("LNPBO_GPU_MIN_FREE_GIB", _DEFAULT_GPU_MIN_FREE_GIB))
        except ValueError:
            min_free = _DEFAULT_GPU_MIN_FREE_GIB
        ok, free_gib = _gpu_has_headroom(min_free)
        if ok:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            fallback_reason = f"only {free_gib:.2f} GiB free on CUDA (need {min_free:.2f})"
    else:
        if use_mps is None:
            use_mps = os.environ.get("LNPBO_USE_MPS", "").lower() in {"1", "true", "yes"}
        if use_mps and torch.backends.mps.is_available():
            warnings.warn(
                "MPS backend uses float32 which may cause numerical instability "
                "in GP fitting (Cholesky failures, poor hyperparameter optimization). "
                "Use CPU or CUDA for reliable results.",
                stacklevel=2,
            )
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    global _DEVICE_LOGGED
    if not _DEVICE_LOGGED:
        _DEVICE_LOGGED = True
        _log_device_selection(device, fallback_reason=fallback_reason)
    return device


def _log_device_selection(device: torch.device, fallback_reason: str = "") -> None:
    """Print a one-line device summary the first time a GP fit picks a device."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_bytes = torch.cuda.get_device_properties(idx).total_memory
        total_gib = total_bytes / (1024**3)
        print(
            f"[GP] Fitting on device=cuda:{idx} ({name}, {total_gib:.1f} GiB)",
            flush=True,
        )
    elif device.type == "mps":
        print("[GP] Fitting on device=mps (Apple Silicon, float32)", flush=True)
    else:
        if fallback_reason:
            print(f"[GP] Fitting on device=cpu (float64) - fell back: {fallback_reason}", flush=True)
        else:
            print("[GP] Fitting on device=cpu (float64)", flush=True)


def _to_tensor(
    X: np.ndarray,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """NumPy -> contiguous torch Tensor on device (float32 on MPS, else float64)."""
    if dtype is None:
        dtype = torch.float32 if device.type == "mps" else torch.float64
    return torch.tensor(np.ascontiguousarray(X), dtype=dtype, device=device)
