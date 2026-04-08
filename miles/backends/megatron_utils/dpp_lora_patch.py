"""Monkey-patch Megatron DDP to place LoRA adapter params in a separate buffer.

When ``offload_train`` is enabled, ``torch_memory_saver.pause()`` offloads all
tracked GPU memory to CPU.  By default, base and LoRA params share the same DDP
flat buffer, so both are offloaded together.  This patch splits adapter params
into their own ``_ParamAndGradBuffer`` with ``disable_param_buffers_cpu_backup=True``,
so that ``pause()`` only offloads the base-weight buffer while adapter weights
remain on GPU.
"""

import logging

import torch
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer, partition_buckets
from megatron.core.process_groups_config import ProcessGroupCollection

from miles.backends.megatron_utils.lora_utils import _is_adapter_param_name

logger = logging.getLogger(__name__)

_patched = False


def patch_ddp_for_colocate_mode_lora() -> None:
    """Monkey-patch Megatron DDP to separate LoRA adapter params into their own buffer.

    In colocate mode with offload_train, torch_memory_saver.pause(tag="default")
    offloads default-region GPU memory.  By default, base and LoRA params share
    the same DDP flat buffer, so both would be affected together.

    This patch ensures adapter params get their own _ParamAndGradBuffer with
    disable_param_buffers_cpu_backup=True (allocated in "param_buffer" region,
    enable_cpu_backup=False).  Since pause(tag="default") only targets the
    "default" region, the adapter buffer is never offloaded and LoRA weights
    remain on GPU across sleep/wake cycles — eliminating the need for
    resume()/pause() around update_weights.

    The patch is idempotent and only takes effect once.
    """
    global _patched
    if _patched:
        return
    _patched = True

    _original_init = DistributedDataParallel.__init__

    def _patched_init(self, *args, **kwargs):
        module = kwargs.get("module") or args[2]
        adapter_param_names: set[str] = set()
        adapter_params: set = set()
        for name, param in module.named_parameters():
            if _is_adapter_param_name(name):
                adapter_param_names.add(name)
                adapter_params.add(param)

        if not adapter_params:
            return _original_init(self, *args, **kwargs)

        # setting requires_grad=False makes DDP skip them during buffer creation
        for p in adapter_params:
            p.requires_grad = False

        _original_init(self, *args, **kwargs)

        # restore requires_grad and create adapter buffers
        for p in adapter_params:
            p.requires_grad = True
            p.grad_added_to_main_grad = False
            self.params_with_grad.append(p)

        param_to_name = {}
        for name, param in module.named_parameters():
            param_to_name[param] = name

        if hasattr(self, "ddp_config") and self.ddp_config.average_in_collective:
            gradient_scaling_factor = 1.0
        else:
            gradient_scaling_factor = 1.0 / self.intra_dp_cp_group.size()

        from megatron.core.fp8_utils import is_float8tensor

        dtype_to_params: dict[tuple, list] = {}
        dtype_to_offsets: dict[tuple, int] = {}
        dtype_to_indices: dict[tuple, list] = {}

        for param in adapter_params:
            if not param.requires_grad:
                continue
            param_dtype = torch.uint8 if is_float8tensor(param) else param.dtype
            grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype

            key = (param_dtype, grad_dtype)
            dtype_to_params.setdefault(key, []).append(param)

            offset_key = (param.dtype, grad_dtype)
            offset = dtype_to_offsets.get(offset_key, 0)
            dtype_to_offsets[offset_key] = offset + 1
            dtype_to_indices.setdefault(key, []).append(offset)

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = self.tp_group
        pg_collection.dp_cp = self.dp_cp_group

        # create lora buffers with disable_grad/param_buffer_cpu_backup to True
        adapter_buffers = []
        for (param_dtype, grad_dtype), params in dtype_to_params.items():
            adapter_buffers.append(
                _ParamAndGradBuffer(
                    self.ddp_config,
                    param_dtype,
                    grad_dtype,
                    params,
                    self.intra_dp_cp_group,
                    self.bucket_size,
                    param_to_name,
                    gradient_scaling_factor,
                    dtype_to_indices[(param_dtype, grad_dtype)],
                    self.ddp_config.nccl_ub,
                    pg_collection,
                    disable_grad_buffers_cpu_backup=True,
                    disable_param_buffers_cpu_backup=True,
                )
            )

        self.buffers.extend(adapter_buffers)

        # Re-partition all buckets into bucket groups.
        all_buffers = self.buffers
        disable_bucketing = self.bucket_size is None
        self.bucket_groups = partition_buckets(all_buffers, force_single_bucket_group=disable_bucketing)

        # rebuild param to bucket_group mapping
        self.param_to_bucket_group = {}
        for bucket_group in self.bucket_groups:
            for bucket in bucket_group.buckets:
                for param in bucket.params_list:
                    self.param_to_bucket_group[param] = bucket_group

        # reset next_param_gather_bucket_group for overlap.
        if self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather:
            num_bucket_groups = len(self.bucket_groups)
            for i in range(1, num_bucket_groups):
                self.bucket_groups[num_bucket_groups - i].next_param_gather_bucket_group = self.bucket_groups[
                    num_bucket_groups - i - 1
                ]

        num_adapter = sum(len(p) for p in dtype_to_params.values())
        logger.info(
            "DDP LoRA patch: created %d separate adapter buffer(s) for %d params "
            "(disable_param_buffers_cpu_backup=True)",
            len(adapter_buffers),
            num_adapter,
        )

    DistributedDataParallel.__init__ = _patched_init
    logger.info("Patched DistributedDataParallel.__init__ for LoRA adapter buffer separation")
