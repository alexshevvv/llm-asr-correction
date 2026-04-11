#!/usr/bin/env python3
"""Memory management helpers for ML workflows.

Used mainly in Colab where VRAM is limited and models
must be unloaded between baseline runs. Safe to call
on machines without CUDA (empty_cache is guarded).
"""

import gc
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def release_model(model: Any) -> None:
    """
    Release references held by a model and free GPU cache.

    Does NOT delete the caller's own reference to the model.
    The caller must follow up with `del model` to fully
    release the object.

    Args:
        model: Any object holding torch tensors.
    """

    for attr in list(vars(model).keys()):
        try:
            delattr(model, attr)
        except Exception:
            pass

    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def report_memory() -> dict[str, float]:
    """
    Return current GPU memory usage in gigabytes.

    Returns:
        Dict with keys allocated_gb, reserved_gb, total_gb.
        All values are 0.0 if CUDA is not available.
    """

    result = {
        'allocated_gb': 0.0,
        'reserved_gb': 0.0,
        'total_gb': 0.0,
    }
    try:
        if not torch.cuda.is_available():
            return result

        result['allocated_gb'] = (
            torch.cuda.memory_allocated() / 1024 ** 3
        )
        result['reserved_gb'] = (
            torch.cuda.memory_reserved() / 1024 ** 3
        )
        result['total_gb'] = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024 ** 3
        )
    except ImportError:
        pass

    return result
