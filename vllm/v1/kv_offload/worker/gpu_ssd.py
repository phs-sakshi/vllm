# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU-SSD Offloading Handler using GPU Direct Storage (GDS).

This handler performs asynchronous data transfers between GPU memory
and SSD storage using NVIDIA's GPU Direct Storage technology via
the kvikio library.
"""
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec, SSDLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)

# Lazy import kvikio to avoid loading if not used
_kvikio_available = None
_cufile = None


def _check_kvikio_available() -> bool:
    """Check if kvikio is available and GDS is supported."""
    global _kvikio_available, _cufile
    if _kvikio_available is not None:
        return _kvikio_available

    try:
        import kvikio
        import kvikio.cufile as cufile

        _cufile = cufile
        _kvikio_available = True

        # Check if GDS is available
        try:
            driver_props = kvikio.DriverProperties()
            if driver_props.is_gds_available:
                logger.info(
                    "GPU Direct Storage (GDS) is available. "
                    "Major version: %d, Minor version: %d",
                    driver_props.major_version,
                    driver_props.minor_version,
                )
            else:
                logger.warning(
                    "GDS driver not available. Falling back to compatibility mode. "
                    "For best performance, install and configure GDS."
                )
        except Exception as e:
            logger.warning("Could not query GDS driver properties: %s", e)

        logger.info("kvikio is available for GPU-SSD transfers")

    except ImportError:
        _kvikio_available = False
        logger.warning(
            "kvikio is not available. Install it with: "
            "pip install kvikio-cu12 (or appropriate CUDA version)"
        )

    return _kvikio_available


def expand_block_ids(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    skip_count: int = 0,
):
    """
    Convert a list of block IDs to a list of matching block ids,
    assuming each block is composed of actual block_size_factor blocks.
    Outputs to output tensor.
    The first skip_count blocks will be skipped.
    Note that skip_count must be less than block_size_factor.

    For example, if block_ids = [0, 1, 3] and block_size_factor = 4,
    then it yields [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]
    since 0 maps to [0, 1, 2, 3]
    1 maps to [4, 5, 6, 7]
    and 3 maps to [12, 13, 14, 15]
    """
    assert skip_count < block_size_factor

    first_range = np.arange(skip_count, block_size_factor)
    full_range = np.arange(0, block_size_factor)

    output_idx = 0
    for i, block_id in enumerate(block_ids):
        base_block_id = block_id * block_size_factor
        indices = first_range if i == 0 else full_range
        output_end_idx = output_idx + len(indices)
        output[output_idx:output_end_idx] = base_block_id + indices
        output_idx = output_end_idx


class GpuSsdOffloadingHandler(OffloadingHandler):
    """Handler for GPU-SSD KV cache transfers using GPU Direct Storage.

    This handler uses kvikio (NVIDIA RAPIDS) to perform direct GPU-to-SSD
    data transfers without going through CPU memory, significantly reducing
    latency and CPU overhead.

    Args:
        gpu_block_size: Block size in tokens for GPU cache.
        ssd_block_size: Block size in tokens for SSD cache.
        num_ssd_blocks: Number of blocks allocated on SSD.
        gpu_caches: Dictionary mapping layer names to GPU KV cache tensors.
        attn_backends: Dictionary mapping layer names to attention backends.
        ssd_cache_dir: Directory path for SSD cache files.
        cache_file_prefix: Prefix for cache file names.
        num_io_threads: Number of threads for async I/O operations.
    """

    def __init__(
        self,
        gpu_block_size: int,
        ssd_block_size: int,
        num_ssd_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        ssd_cache_dir: str,
        cache_file_prefix: str = "kv_cache",
        num_io_threads: int = 4,
    ):
        if not _check_kvikio_available():
            raise RuntimeError(
                "kvikio is required for SSD offloading. "
                "Install it with: pip install kvikio-cu12"
            )

        assert ssd_block_size % gpu_block_size == 0
        self.block_size_factor = ssd_block_size // gpu_block_size

        # Thread pool for async I/O operations
        self.io_executor = ThreadPoolExecutor(
            max_workers=num_io_threads, thread_name_prefix="ssd_io"
        )

        # Track pending transfers: job_id -> Future
        self.pending_transfers: dict[int, Future] = {}

        # Storage configuration
        self.ssd_cache_dir = Path(ssd_cache_dir)
        self.cache_file_prefix = cache_file_prefix
        self.num_ssd_blocks = num_ssd_blocks

        # Process GPU caches and create SSD cache files
        logger.info("Setting up SSD cache for %d GPU tensors...", len(gpu_caches))
        self.gpu_tensors: list[torch.Tensor] = []
        self.kv_dim_before_num_blocks: list[bool] = []
        self.block_sizes_bytes: list[int] = []
        self.ssd_files: list["_cufile.CuFile"] = []

        for layer_idx, (layer_name, gpu_tensor) in enumerate(gpu_caches.items()):
            self.gpu_tensors.append(gpu_tensor)

            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            if len(gpu_shape) != len(test_shape):
                # cross-layers tensor
                # shape is (num_blocks, ...)
                assert len(gpu_shape) == len(test_shape) + 1
                num_blocks_idx = 0
                self.kv_dim_before_num_blocks.append(False)
            elif test_shape[0] == 1234:
                # shape is (num_blocks, ...)
                num_blocks_idx = 0
                self.kv_dim_before_num_blocks.append(False)
            else:
                # shape should be (2, num_blocks, ...)
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2
                num_blocks_idx = 1
                self.kv_dim_before_num_blocks.append(True)

            # Calculate block size in bytes
            # For a single block, we need to figure out the memory footprint
            block_shape = list(gpu_shape)
            block_shape[num_blocks_idx] = 1  # Single block
            single_block_elements = 1
            for dim in block_shape:
                single_block_elements *= dim
            block_size_bytes = single_block_elements * gpu_tensor.element_size()

            # Account for block_size_factor (SSD block = multiple GPU blocks)
            ssd_block_size_bytes = block_size_bytes * self.block_size_factor
            self.block_sizes_bytes.append(ssd_block_size_bytes)

            # Create cache file for this layer
            cache_file_path = (
                self.ssd_cache_dir / f"{cache_file_prefix}_layer_{layer_idx}.bin"
            )

            # Pre-allocate the cache file
            total_size = ssd_block_size_bytes * num_ssd_blocks
            self._create_cache_file(cache_file_path, total_size)

            # Open with kvikio for GDS access
            ssd_file = _cufile.CuFile(cache_file_path, "r+")
            self.ssd_files.append(ssd_file)

            logger.debug(
                "Layer %d: block_size=%d bytes, total=%d bytes, file=%s",
                layer_idx,
                ssd_block_size_bytes,
                total_size,
                cache_file_path,
            )

    def _create_cache_file(self, path: Path, size: int):
        """Create a pre-allocated cache file for GDS.

        Args:
            path: Path to the cache file.
            size: Total size of the file in bytes.
        """
        # Use fallocate for efficient pre-allocation if available
        try:
            with open(path, "wb") as f:
                # Pre-allocate file with fallocate (Linux)
                try:
                    os.posix_fallocate(f.fileno(), 0, size)
                except (AttributeError, OSError):
                    # Fallback: write zeros
                    f.seek(size - 1)
                    f.write(b"\0")
            logger.debug("Created cache file %s with size %d bytes", path, size)
        except Exception as e:
            logger.error("Failed to create cache file %s: %s", path, e)
            raise

    def _get_block_slice(
        self, tensor: torch.Tensor, block_ids: np.ndarray, kv_dim_before: bool
    ) -> torch.Tensor:
        """Get a view of the tensor containing the specified blocks.

        Args:
            tensor: The KV cache tensor.
            block_ids: Array of block IDs to extract.
            kv_dim_before: Whether KV dimension comes before num_blocks.

        Returns:
            View of the tensor containing the specified blocks.
        """
        if kv_dim_before:
            # Shape: (2, num_blocks, ...)
            # Get both K and V for the specified blocks
            return tensor[:, block_ids, ...]
        else:
            # Shape: (num_blocks, ...) or (num_blocks, 2, ...)
            return tensor[block_ids, ...]

    def _do_gpu_to_ssd_transfer(
        self,
        src_block_ids: np.ndarray,
        dst_block_ids: np.ndarray,
    ) -> bool:
        """Perform GPU to SSD transfer for specified blocks.

        Args:
            src_block_ids: Source block IDs in GPU memory.
            dst_block_ids: Destination block IDs in SSD storage.

        Returns:
            True if transfer was successful.
        """
        try:
            for layer_idx, (gpu_tensor, ssd_file, block_size, kv_dim) in enumerate(
                zip(
                    self.gpu_tensors,
                    self.ssd_files,
                    self.block_sizes_bytes,
                    self.kv_dim_before_num_blocks,
                )
            ):
                for src_block, dst_block in zip(src_block_ids, dst_block_ids):
                    # Calculate file offset for destination block
                    # Convert to Python int for kvikio compatibility
                    src_idx = int(src_block)
                    file_offset = int(dst_block) * block_size

                    # Get the source block data from GPU
                    if kv_dim:
                        # For (2, num_blocks, ...) layout
                        k_data = gpu_tensor[0, src_idx : src_idx + 1, ...]
                        v_data = gpu_tensor[1, src_idx : src_idx + 1, ...]
                        # Stack K and V together for single write
                        block_data = torch.cat([k_data, v_data], dim=0).contiguous()
                    else:
                        block_data = gpu_tensor[
                            src_idx : src_idx + 1, ...
                        ].contiguous()

                    # Write to SSD using GDS
                    ssd_file.pwrite(block_data, file_offset)

            return True
        except Exception as e:
            logger.error("GPU to SSD transfer failed: %s", e)
            return False

    def _do_ssd_to_gpu_transfer(
        self,
        src_block_ids: np.ndarray,
        dst_block_ids: np.ndarray,
    ) -> bool:
        """Perform SSD to GPU transfer for specified blocks.

        Args:
            src_block_ids: Source block IDs in SSD storage.
            dst_block_ids: Destination block IDs in GPU memory.

        Returns:
            True if transfer was successful.
        """
        try:
            for layer_idx, (gpu_tensor, ssd_file, block_size, kv_dim) in enumerate(
                zip(
                    self.gpu_tensors,
                    self.ssd_files,
                    self.block_sizes_bytes,
                    self.kv_dim_before_num_blocks,
                )
            ):
                for src_block, dst_block in zip(src_block_ids, dst_block_ids):
                    # Calculate file offset for source block
                    # Convert to Python int for kvikio compatibility
                    dst_idx = int(dst_block)
                    file_offset = int(src_block) * block_size

                    if kv_dim:
                        # For (2, num_blocks, ...) layout, read K and V
                        k_data = gpu_tensor[0, dst_idx : dst_idx + 1, ...]
                        v_data = gpu_tensor[1, dst_idx : dst_idx + 1, ...]

                        # Create a temporary buffer for reading
                        block_data = torch.cat(
                            [k_data.clone(), v_data.clone()], dim=0
                        ).contiguous()

                        # Read from SSD using GDS
                        ssd_file.pread(block_data, file_offset)

                        # Split and copy back to K and V caches
                        k_data.copy_(block_data[0:1, ...])
                        v_data.copy_(block_data[1:2, ...])
                    else:
                        # Direct read into GPU tensor
                        dst_slice = gpu_tensor[dst_idx : dst_idx + 1, ...]
                        ssd_file.pread(dst_slice.contiguous(), file_offset)

            return True
        except Exception as e:
            logger.error("SSD to GPU transfer failed: %s", e)
            return False

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """Initiate an asynchronous transfer.

        Args:
            job_id: Unique ID for tracking this transfer.
            spec: Transfer specification (source, destination).

        Returns:
            True if transfer was successfully initiated.
        """
        src_spec, dst_spec = spec

        if isinstance(src_spec, SSDLoadStoreSpec):
            assert isinstance(dst_spec, GPULoadStoreSpec)
            is_read = True
        else:
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, SSDLoadStoreSpec)
            is_read = False

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        # Handle block size factor conversion
        if is_read:
            # SSD -> GPU: SSD blocks are larger
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
        else:
            # GPU -> SSD: GPU blocks are smaller
            src_block_size_factor = 1
            dst_block_size_factor = self.block_size_factor

        src_sub_block_count = src_blocks.size * src_block_size_factor
        dst_sub_block_count = dst_blocks.size * dst_block_size_factor
        src_sub_blocks_to_skip = -dst_blocks.size % src_block_size_factor

        assert dst_sub_block_count == src_sub_block_count - src_sub_blocks_to_skip

        # Expand block IDs for the transfer
        expanded_src = np.empty(dst_sub_block_count, dtype=np.int64)
        expanded_dst = np.empty(dst_sub_block_count, dtype=np.int64)
        expand_block_ids(
            src_blocks, src_block_size_factor, expanded_src, skip_count=src_sub_blocks_to_skip
        )
        expand_block_ids(dst_blocks, dst_block_size_factor, expanded_dst)

        # Submit async transfer
        if is_read:
            future = self.io_executor.submit(
                self._do_ssd_to_gpu_transfer, expanded_src, expanded_dst
            )
        else:
            future = self.io_executor.submit(
                self._do_gpu_to_ssd_transfer, expanded_src, expanded_dst
            )

        self.pending_transfers[job_id] = future
        return True

    def get_finished(self) -> list[TransferResult]:
        """Get list of completed transfers.

        Returns:
            List of (job_id, success) tuples for completed transfers.
        """
        results: list[TransferResult] = []
        completed_jobs = []

        for job_id, future in self.pending_transfers.items():
            if future.done():
                try:
                    success = future.result()
                except Exception as e:
                    logger.error("Transfer job %d failed with exception: %s", job_id, e)
                    success = False
                results.append((job_id, success))
                completed_jobs.append(job_id)

        for job_id in completed_jobs:
            del self.pending_transfers[job_id]

        return results

    def close(self):
        """Clean up resources."""
        # Wait for pending transfers
        for future in self.pending_transfers.values():
            try:
                future.result(timeout=30)
            except Exception as e:
                logger.warning("Transfer did not complete during cleanup: %s", e)

        # Close SSD files
        for ssd_file in self.ssd_files:
            try:
                ssd_file.close()
            except Exception as e:
                logger.warning("Failed to close SSD file: %s", e)

        # Shutdown thread pool
        self.io_executor.shutdown(wait=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
