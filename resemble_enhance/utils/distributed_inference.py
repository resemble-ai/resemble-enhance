"""
Simplified distributed utilities for inference without DeepSpeed dependencies.
This module provides no-op versions of distributed functions for inference use.
"""


def global_leader_only(fn):
    """No-op decorator for inference - just returns the function as-is."""
    return fn


def local_leader_only(fn):
    """No-op decorator for inference - just returns the function as-is."""
    return fn


def is_global_leader():
    """Always return True for inference (single process)."""
    return True


def is_local_leader():
    """Always return True for inference (single process)."""
    return True
