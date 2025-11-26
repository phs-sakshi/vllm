# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SSD Offloading Spec for KV cache offloading using GPU Direct Storage (GDS).

This module provides the SSDOffloadingSpec class which configures and manages
KV cache offloading from GPU to SSD using NVIDIA's GPU Direct Storage
technology for efficient data transfers.

Example usage:
    vllm serve <model> \
        --kv-offloading-backend native \
        --kv-offloading-size 100 \
        --kv-connector-extra-config '{
            "spec_name": "SSDOffloadingSpec",
            "num_ssd_blocks": 10000,
            "ssd_cache_dir": "/mnt/nvme/vllm_cache",
            "eviction_policy": "lru"
        }'
"""
import os
import tempfile
from collections.abc import Iterator

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.ssd import SSDBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec, SSDLoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.gpu_ssd import GpuSsdOffloadingHandler
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)


class SSDOffloadingSpec(OffloadingSpec):
    """Specification for SSD-based KV cache offloading using GPU Direct Storage.

    This spec enables offloading KV cache from GPU memory to SSD storage
    using NVIDIA's GPU Direct Storage (GDS) technology. GDS allows direct
    data transfer between GPU memory and NVMe SSDs without CPU involvement,
    providing significantly lower latency and higher throughput compared
    to traditional CPU-mediated transfers.

    Requirements:
        - NVIDIA GPU with compute capability 7.0+ (Volta or newer)
        - NVMe SSD with GDS-compatible filesystem (ext4, XFS)
        - kvikio library (pip install kvikio-cu12)
        - CUDA 11.4+ with GDS support

    Configuration options (via kv_connector_extra_config):
        num_ssd_blocks (int): Number of KV cache blocks to allocate on SSD.
            Required.
        ssd_cache_dir (str): Directory path for SSD cache files. Should be
            on a fast NVMe SSD. Defaults to system temp directory.
        cache_file_prefix (str): Prefix for cache file names.
            Defaults to "kv_cache".
        eviction_policy (str): Cache eviction policy, either "lru" or "arc".
            Defaults to "lru".
        num_io_threads (int): Number of I/O threads for async operations.
            Defaults to 4.
        block_size (int): Offloaded block size in tokens. Defaults to GPU
            block size. Should be a multiple of GPU block size.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

        # Get configuration from extra_config
        num_ssd_blocks = self.extra_config.get("num_ssd_blocks")
        if not num_ssd_blocks:
            raise ValueError(
                "num_ssd_blocks must be specified in kv_connector_extra_config. "
                "Example: --kv-connector-extra-config "
                '\'{"spec_name": "SSDOffloadingSpec", "num_ssd_blocks": 10000, '
                '"ssd_cache_dir": "/mnt/nvme/vllm_cache"}\''
            )
        self.num_ssd_blocks: int = num_ssd_blocks

        # SSD cache directory - default to temp directory if not specified
        self.ssd_cache_dir: str = self.extra_config.get(
            "ssd_cache_dir", os.path.join(tempfile.gettempdir(), "vllm_ssd_cache")
        )

        # Cache file prefix
        self.cache_file_prefix: str = self.extra_config.get(
            "cache_file_prefix", "kv_cache"
        )

        # Eviction policy
        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

        # Number of I/O threads
        self.num_io_threads: int = self.extra_config.get("num_io_threads", 4)

        # Scheduler-side manager
        self._manager: OffloadingManager | None = None

        # Worker-side handler
        self._handler: OffloadingHandler | None = None

        logger.info(
            "SSD Offloading configured: %d blocks, dir=%s, policy=%s",
            self.num_ssd_blocks,
            self.ssd_cache_dir,
            self.eviction_policy,
        )

    def get_manager(self) -> OffloadingManager:
        """Get the offloading manager for scheduler-side operations.

        Returns:
            An OffloadingManager instance configured for SSD offloading.
        """
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )

            backend = SSDBackend(
                block_size=self.offloaded_block_size,
                num_blocks=self.num_ssd_blocks,
                ssd_cache_dir=self.ssd_cache_dir,
                cache_file_prefix=self.cache_file_prefix,
            )

            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            else:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}. "
                    f"Supported policies: lru, arc"
                )

        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        """Get offloading handlers for GPU-SSD transfers.

        Args:
            kv_caches: Dictionary mapping layer names to GPU KV cache tensors.
            attn_backends: Dictionary mapping layer names to attention backends.

        Yields:
            Tuples of (src_type, dst_type, handler) for both directions:
            - GPU -> SSD (for offloading)
            - SSD -> GPU (for loading)
        """
        if not self._handler:
            if not current_platform.is_cuda_alike():
                raise RuntimeError(
                    "SSD Offloading with GDS is only supported on CUDA-alike GPUs. "
                    f"Current platform: {current_platform}"
                )

            self._handler = GpuSsdOffloadingHandler(
                gpu_block_size=self.gpu_block_size,
                ssd_block_size=self.offloaded_block_size,
                num_ssd_blocks=self.num_ssd_blocks,
                gpu_caches=kv_caches,
                attn_backends=attn_backends,
                ssd_cache_dir=self.ssd_cache_dir,
                cache_file_prefix=self.cache_file_prefix,
                num_io_threads=self.num_io_threads,
            )

        assert self._handler is not None

        # Yield handlers for both transfer directions
        yield GPULoadStoreSpec, SSDLoadStoreSpec, self._handler  # GPU -> SSD
        yield SSDLoadStoreSpec, GPULoadStoreSpec, self._handler  # SSD -> GPU
