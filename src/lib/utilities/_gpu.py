"""GPU-memory helpers for admission-gated / dynamically-sized worker pools.

``gpu_free_gb`` is the single live-VRAM query; ``size_workers`` is the ``dynamic_n.py`` sizing formula
(``floor((free - reserve) / peak)``) lifted into the library so both the work-pool's ``GpuGate`` and any
caller share one implementation. Both degrade gracefully (return ``None`` / the fallback) when torch or
CUDA is unavailable, so they are safe to import and call on a CPU-only box or under a mocked test.
"""

from __future__ import annotations

from loguru import logger


def gpu_free_gb(device: int = 0) -> float | None:
    """Live free VRAM (GB) on ``device`` via ``torch.cuda.mem_get_info``.

    Returns ``None`` when torch is missing, CUDA is unavailable, or the query raises — the caller then
    treats "unknown free memory" as "no gating" rather than crashing. NB: a successful call creates a
    CUDA context in *this* process (~0.3-0.6 GB pinned); it shows up as consumed in the reading, so a
    gate built on this self-accounts, but a memory budget should reserve headroom for it.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, _total = torch.cuda.mem_get_info(device)
        return free_bytes / 1e9
    except Exception as e:  # noqa: BLE001 — any failure -> "unknown", not a crash
        logger.debug(f"gpu_free_gb: mem query failed ({e})")
        return None


def size_workers(*, peak_gb: float, reserve_gb: float, cap: int, fallback: int) -> int:
    """``floor((free - reserve) / peak)`` clamped to ``[1, cap]``; ``fallback`` when free mem is unknown.

    The number of same-GPU workers whose combined peak fits alongside a ``reserve_gb`` cushion. Mirrors
    ``scripts/dynamic_n.py`` (kept as a thin CLI); the reserve is deliberately generous because a
    just-launched worker's allocation ramp is not yet reflected in an instantaneous free-mem reading.
    """
    free = gpu_free_gb()
    if free is None:
        return fallback
    n = int((free - reserve_gb) // peak_gb)
    return max(1, min(cap, n))
