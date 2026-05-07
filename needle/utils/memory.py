"""GPU memory profiling helpers."""
from __future__ import annotations

from typing import Dict


def gpu_memory_stats(device: int = 0) -> Dict[str, float]:
    """Return GPU memory stats in GB."""
    try:
        import torch
        props = torch.cuda.get_device_properties(device)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        return {
            "total_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(total - reserved, 2),
        }
    except Exception:
        return {}


def print_gpu_memory(device: int = 0) -> None:
    stats = gpu_memory_stats(device)
    if stats:
        print(
            f"GPU:{device} | total={stats['total_gb']}GB | "
            f"alloc={stats['allocated_gb']}GB | free={stats['free_gb']}GB"
        )
