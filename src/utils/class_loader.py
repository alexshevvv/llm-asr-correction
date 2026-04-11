#!/usr/bin/env python3
"""Dotted-path class resolver for registry-driven construction."""

from importlib import import_module


def resolve_class(class_path: str) -> type:
    """
    Resolve a dotted class path to a class object.

    Args:
        class_path: Full dotted path, e.g. 'src.asr.base.BaseASR'.

    Returns:
        The class object.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
    """

    module_path, class_name = class_path.rsplit('.', 1)
    module = import_module(module_path)
    return getattr(module, class_name)
