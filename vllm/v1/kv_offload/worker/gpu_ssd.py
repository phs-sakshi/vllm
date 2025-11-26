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
_kvikio = None

# CUDA memcpy helper using ctypes
_cudart = None


def _cuda_memcpy(dst_ptr: int, src_ptr: int, size: int) -> None:
    """
    Perform a CUDA device-to-device memory copy using cudaMemcpy.
    This bypasses PyTorch's inference mode restrictions.
    """
    global _cudart
    if _cudart is None:
        import ctypes
        # Load CUDA runtime library
        try:
            _cudart = ctypes.CDLL("libcudart.so")
        except OSError:
            # Try with version suffix
            import ctypes.util
            cudart_path = ctypes.util.find_library("cudart")
            if cudart_path:
                _cudart = ctypes.CDLL(cudart_path)
            else:
                raise RuntimeError("Could not find CUDA runtime library")
        
        # Set up cudaMemcpy signature
        # cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
        _cudart.cudaMemcpy.argtypes = [
            ctypes.c_void_p,  # dst
            ctypes.c_void_p,  # src
            ctypes.c_size_t,  # count
            ctypes.c_int,     # kind
        ]
        _cudart.cudaMemcpy.restype = ctypes.c_int
    
    # cudaMemcpyDeviceToDevice = 3
    cudaMemcpyDeviceToDevice = 3
    result = _cudart.cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice)
    if result != 0:
        raise RuntimeError(f"cudaMemcpy failed with error code {result}")


def _check_kvikio_available() -> bool:
    """Check if kvikio is available and GDS is supported."""
    global _kvikio_available, _kvikio
    if _kvikio_available is not None:
        return _kvikio_available

    try:
        import kvikio

        _kvikio = kvikio
        _kvikio_available = True

        # Check if GDS is available
        try:
            if hasattr(kvikio, 'DriverProperties'):
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
                        "GDS driver not available. Falling back to compatibility mode."
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
    """Handler for GPU-SSD KV cache transfers using GPU Direct Storage."""

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
        self.gpu_block_size = gpu_block_size

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
        self.ssd_file_paths: list[Path] = []

        for layer_idx, (layer_name, gpu_tensor) in enumerate(gpu_caches.items()):
            self.gpu_tensors.append(gpu_tensor)

            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            if len(gpu_shape) != len(test_shape):
                num_blocks_idx = 0
                self.kv_dim_before_num_blocks.append(False)
            elif test_shape[0] == 1234:
                num_blocks_idx = 0
                self.kv_dim_before_num_blocks.append(False)
            else:
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2
                num_blocks_idx = 1
                self.kv_dim_before_num_blocks.append(True)

            # Calculate block size in bytes for a single GPU block
            block_shape = list(gpu_shape)
            block_shape[num_blocks_idx] = 1
            single_block_elements = 1
            for dim in block_shape:
                single_block_elements *= dim
            gpu_block_size_bytes = single_block_elements * gpu_tensor.element_size()
            self.block_sizes_bytes.append(gpu_block_size_bytes)

            # Create cache file for this layer
            cache_file_path = (
                self.ssd_cache_dir / f"{cache_file_prefix}_layer_{layer_idx}.bin"
            )
            self.ssd_file_paths.append(cache_file_path)

            # Pre-allocate the cache file - need space for all SSD blocks
            # Each SSD block = block_size_factor GPU blocks
            total_size = gpu_block_size_bytes * self.block_size_factor * num_ssd_blocks
            max_gpu_block_index = self.block_size_factor * num_ssd_blocks - 1
            self._create_cache_file(cache_file_path, total_size)

            logger.info(
                "Layer %d: gpu_shape=%s, block_size_bytes=%d, total_size=%d bytes, "
                "block_size_factor=%d, max_gpu_block_idx=%d, file=%s",
                layer_idx,
                list(gpu_shape),
                gpu_block_size_bytes,
                total_size,
                self.block_size_factor,
                max_gpu_block_index,
                cache_file_path,
            )

    def _create_cache_file(self, path: Path, size: int):
        """Create a pre-allocated cache file for GDS."""
        try:
            with open(path, "wb") as f:
                try:
                    os.posix_fallocate(f.fileno(), 0, size)
                except (AttributeError, OSError):
                    f.seek(size - 1)
                    f.write(b"\0")
            # Verify file was created with correct size
            actual_size = os.path.getsize(path)
            logger.info(
                "Created cache file %s: requested=%d bytes, actual=%d bytes",
                path, size, actual_size
            )
            if actual_size != size:
                raise RuntimeError(
                    f"File size mismatch: expected {size}, got {actual_size}"
                )
        except Exception as e:
            logger.error("Failed to create cache file %s: %s", path, e)
            raise

    def _do_gpu_to_ssd_transfer(
        self,
        src_block_ids: np.ndarray,
        dst_block_ids: np.ndarray,
    ) -> bool:
        """Perform GPU to SSD transfer for specified blocks."""
        try:
            for layer_idx, (gpu_tensor, file_path, block_size_bytes, kv_dim) in enumerate(
                zip(
                    self.gpu_tensors,
                    self.ssd_file_paths,
                    self.block_sizes_bytes,
                    self.kv_dim_before_num_blocks,
                )
            ):
                file_size = os.path.getsize(file_path)
                logger.debug(
                    "GPU→SSD layer %d: file_size=%d, block_size_bytes=%d, "
                    "src_blocks=%s, dst_blocks=%s",
                    layer_idx, file_size, block_size_bytes,
                    src_block_ids.tolist(), dst_block_ids.tolist()
                )

                # Open file for this transfer
                with _kvikio.CuFile(file_path, "r+") as ssd_file:
                    for src_block, dst_block in zip(src_block_ids, dst_block_ids):
                        src_idx = int(src_block)
                        dst_idx = int(dst_block)
                        file_offset = dst_idx * block_size_bytes

                        # Get the source block data from GPU and make a contiguous copy
                        if kv_dim:
                            k_data = gpu_tensor[0, src_idx, ...].contiguous().clone()
                            v_data = gpu_tensor[1, src_idx, ...].contiguous().clone()
                            block_data = torch.cat(
                                [k_data.unsqueeze(0), v_data.unsqueeze(0)], dim=0
                            ).contiguous()
                        else:
                            block_data = gpu_tensor[src_idx, ...].contiguous().clone()

                        # Validate offset before write
                        end_offset = file_offset + block_data.nbytes
                        if end_offset > file_size:
                            raise RuntimeError(
                                f"Write would exceed file size: offset={file_offset}, "
                                f"size={block_data.nbytes}, end={end_offset}, "
                                f"file_size={file_size}, dst_idx={dst_idx}"
                            )

                        logger.debug(
                            "Writing block: src=%d, dst=%d, offset=%d, size=%d",
                            src_idx, dst_idx, file_offset, block_data.nbytes
                        )

                        # Write to SSD using GDS - pwrite returns IOFuture, call .get() to wait
                        # NOTE: Must use keyword args! pwrite(buf, size, file_offset, ...)
                        future = ssd_file.pwrite(block_data, file_offset=file_offset)
                        nbytes = future.get()  # Wait for I/O to complete
                        if nbytes != block_data.nbytes:
                            raise RuntimeError(
                                f"Incomplete write: expected {block_data.nbytes}, got {nbytes}"
                            )

            return True
        except Exception as e:
            logger.error("GPU to SSD transfer failed: %s", e)
            return False

    def _do_ssd_to_gpu_transfer(
        self,
        src_block_ids: np.ndarray,
        dst_block_ids: np.ndarray,
    ) -> bool:
        """Perform SSD to GPU transfer for specified blocks."""
        try:
            for layer_idx, (gpu_tensor, file_path, block_size_bytes, kv_dim) in enumerate(
                zip(
                    self.gpu_tensors,
                    self.ssd_file_paths,
                    self.block_sizes_bytes,
                    self.kv_dim_before_num_blocks,
                )
            ):
                # Open file for this transfer
                with _kvikio.CuFile(file_path, "r") as ssd_file:
                    for src_block, dst_block in zip(src_block_ids, dst_block_ids):
                        src_idx = int(src_block)
                        dst_idx = int(dst_block)
                        file_offset = src_idx * block_size_bytes

                        if kv_dim:
                            # Create buffer matching the file layout
                            k_shape = gpu_tensor[0, dst_idx, ...].shape
                            buffer = torch.empty(
                                (2,) + k_shape,
                                dtype=gpu_tensor.dtype,
                                device=gpu_tensor.device,
                            )

                            # Read from SSD - pread returns IOFuture, call .get() to wait
                            # NOTE: Must use keyword args! pread(buf, size, file_offset, ...)
                            future = ssd_file.pread(buffer, file_offset=file_offset)
                            nbytes = future.get()  # Wait for I/O to complete
                            if nbytes != buffer.nbytes:
                                raise RuntimeError(
                                    f"Incomplete read: expected {buffer.nbytes}, got {nbytes}"
                                )

                            # Copy to GPU cache using direct CUDA memcpy to bypass inference mode
                            _cuda_memcpy(
                                gpu_tensor[0, dst_idx, ...].data_ptr(),
                                buffer[0].data_ptr(),
                                buffer[0].nbytes,
                            )
                            _cuda_memcpy(
                                gpu_tensor[1, dst_idx, ...].data_ptr(),
                                buffer[1].data_ptr(),
                                buffer[1].nbytes,
                            )
                        else:
                            # Create buffer for reading
                            block_shape = gpu_tensor[dst_idx, ...].shape
                            buffer = torch.empty(
                                block_shape,
                                dtype=gpu_tensor.dtype,
                                device=gpu_tensor.device,
                            )

                            # Read from SSD - pread returns IOFuture, call .get() to wait
                            # NOTE: Must use keyword args! pread(buf, size, file_offset, ...)
                            future = ssd_file.pread(buffer, file_offset=file_offset)
                            nbytes = future.get()  # Wait for I/O to complete
                            if nbytes != buffer.nbytes:
                                raise RuntimeError(
                                    f"Incomplete read: expected {buffer.nbytes}, got {nbytes}"
                                )

                            # Copy to GPU cache using direct CUDA memcpy to bypass inference mode
                            _cuda_memcpy(
                                gpu_tensor[dst_idx, ...].data_ptr(),
                                buffer.data_ptr(),
                                buffer.nbytes,
                            )

            return True
        except Exception as e:
            logger.error("SSD to GPU transfer failed: %s", e)
            return False

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """Initiate an asynchronous transfer."""
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

        logger.debug(
            "transfer_async job=%d: is_read=%s, src_blocks=%s, dst_blocks=%s, "
            "block_size_factor=%d, num_ssd_blocks=%d",
            job_id, is_read, src_blocks.tolist(), dst_blocks.tolist(),
            self.block_size_factor, self.num_ssd_blocks
        )

        # Handle block size factor conversion
        if is_read:
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
        else:
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

        logger.debug(
            "Expanded: src=%s, dst=%s, max_dst=%d",
            expanded_src.tolist(), expanded_dst.tolist(),
            int(np.max(expanded_dst)) if len(expanded_dst) > 0 else -1
        )

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
        """Get list of completed transfers."""
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
        for future in self.pending_transfers.values():
            try:
                future.result(timeout=30)
            except Exception as e:
                logger.warning("Transfer did not complete during cleanup: %s", e)

        self.io_executor.shutdown(wait=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
