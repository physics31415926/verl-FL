# Copyright (c) 2026 BAAI. All rights reserved.
# Re-export PyFlagcxCommunicator from vllm-plugin-FL

"""
FlagCX communicator for cross-device weight synchronization.

This module re-exports PyFlagcxCommunicator from vllm-plugin-FL to avoid
code duplication. The vllm-plugin-FL version provides a complete implementation
with all collective operations (broadcast, all_reduce, send/recv, etc.).
"""

from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator  # noqa: F401

__all__ = ["PyFlagcxCommunicator"]
