# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for distributed training."""

import ctypes
import logging
import os
from datetime import timedelta

import ray
import torch.distributed

from verl.utils.device import get_device_name, get_dist_backend, get_nccl_backend, get_torch_device, is_npu_available

logger = logging.getLogger(__name__)


def set_numa_affinity():
    if is_npu_available:
        # TODO (FightingZhen) libnuma.so is not available in e2e_ascend CI image, remove this code after image update.
        return

    initialized = False
    try:
        libnuma = ctypes.CDLL("libnuma.so")
        if libnuma.numa_available() < 0:
            return

        import pynvml

        pynvml.nvmlInit()
        initialized = True
        device_name = "NPU" if is_npu_available else "GPU"
        local_rank = int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        pynvml.nvmlDeviceSetCpuAffinity(handle)
    except ImportError:
        print("Warning: pynvml not available, skipping NUMA affinity setup")
    except Exception as e:
        print(f"Warning: Failed to set NUMA affinity: {e}")
    finally:
        if initialized:
            pynvml.nvmlShutdown()


def initialize_global_process_group(timeout_second=36000):
    torch.distributed.init_process_group(
        get_dist_backend(),
        timeout=timedelta(seconds=timeout_second),
        init_method=os.environ.get("DIST_INIT_METHOD", None),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        get_torch_device().set_device(local_rank)
    return local_rank, rank, world_size


def destroy_global_process_group():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def initialize_global_process_group_ray(timeout_second=None):
    # in current ray environment, LOCAL_RANK is always zero.

    import torch.distributed

    timeout = timedelta(seconds=timeout_second) if timeout_second is not None else None

    if not torch.distributed.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.distributed.init_process_group(
            backend=get_dist_backend(),
            rank=rank,
            world_size=world_size,
            timeout=timeout,
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )


def vllm_stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """Create a stateless communicator for weight synchronization with vLLM workers.

    Uses vLLM's ``StatelessProcessGroup`` for TCP rendezvous, then initialises
    the appropriate data-plane communicator based on the platform backend:
    - FlagCX: :class:`~verl.utils.flagcx_communicator.PyFlagcxCommunicator`
    - NPU (Ascend): ``PyHcclCommunicator`` from vllm_ascend
    - CUDA: ``PyNcclCommunicator`` from vllm
    """
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)

    comm_backend = get_nccl_backend()
    logger.info(
        "vllm_stateless_init_process_group: backend=%s, rank=%d, world_size=%d, device=%s",
        comm_backend,
        rank,
        world_size,
        device,
    )

    if comm_backend == "flagcx":
        from verl.utils.flagcx_communicator import PyFlagcxCommunicator

        # Convert int device to device string (e.g., 0 -> "musa:0" or "cuda:0")
        if isinstance(device, int):
            device_name = get_device_name()
            device = f"{device_name}:{device}"
        return PyFlagcxCommunicator(pg, device=device)
    elif is_npu_available:
        from vllm_ascend.distributed.device_communicators.pyhccl import (
            PyHcclCommunicator as PyNcclCommunicator,
        )
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    return PyNcclCommunicator(pg, device=device)
