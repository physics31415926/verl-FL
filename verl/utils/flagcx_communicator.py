# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagCX communicator for cross-device weight synchronization.

This module re-exports PyFlagcxCommunicator from vllm-plugin-FL to avoid
code duplication. The vllm-plugin-FL version provides a complete implementation
with all collective operations (broadcast, all_reduce, send/recv, etc.).

Note: MUSA Stream compatibility (cuda_stream property) is handled by
PlatformMUSA.ensure_initialized() in verl/plugin/platform/platform_musa.py.
"""

from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator  # noqa: F401

__all__ = ["PyFlagcxCommunicator"]
