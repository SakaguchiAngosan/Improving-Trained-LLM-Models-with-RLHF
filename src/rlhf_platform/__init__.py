"""
rlhf_platform: Production-grade distributed RLHF alignment engine.
Designed for multi-node GPU clusters with asymmetric model parallelism.
"""

__version__ = "0.1.0"
__author__ = "RLHF Infrastructure Team"

from . import alignment, distributed, utils

__all__ = ["alignment", "distributed", "utils"]
