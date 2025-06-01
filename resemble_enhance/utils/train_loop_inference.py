"""
Simplified TrainLoop utilities for inference without training dependencies.
This module provides no-op versions of TrainLoop functions for inference use.
"""


class TrainLoop:
    """No-op TrainLoop class for inference."""
    
    @staticmethod
    def get_running_loop():
        """Always return None for inference (no training loop)."""
        return None
