# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import logging

from verl.utils.device import get_device_name, get_nccl_backend, is_npu_available

logger = logging.getLogger(__name__)


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
