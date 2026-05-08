# Copyright (c) 2026 BAAI. All rights reserved.
# Re-export PyFlagcxCommunicator from vllm-plugin-FL

"""
FlagCX communicator for cross-device weight synchronization.

This module re-exports PyFlagcxCommunicator from vllm-plugin-FL to avoid
code duplication. The vllm-plugin-FL version provides a complete implementation
with all collective operations (broadcast, all_reduce, send/recv, etc.).

Additionally, it patches MUSA Stream objects to expose a `cuda_stream` attribute
so that FlagCX's `adaptor_stream_copy` (which hardcodes `cuda_stream`) works
transparently on MUSA devices.
"""

import torch

from verl.utils.device import is_musa_available

if is_musa_available:
    import torch_musa  # noqa: F401 — ensures torch.musa is populated

    # FlagCX wrapper's adaptor_stream_copy accesses stream.cuda_stream,
    # but MUSA streams use musa_stream. Patch the class to add compatibility.
    _StreamCls = torch.musa.Stream if hasattr(torch, "musa") else None
    if _StreamCls is not None and not hasattr(_StreamCls, "cuda_stream"):
        _StreamCls.cuda_stream = property(lambda self: self.musa_stream)

from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator  # noqa: F401

__all__ = ["PyFlagcxCommunicator"]
