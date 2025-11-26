# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for GpuSsdOffloadingHandler.

These tests verify the GPU-SSD transfer handler works correctly.
Tests require both CUDA and kvikio to be available.
"""
import random
import tempfile
import time

import pytest
import torch

from vllm.platforms import current_platform

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Check if kvikio is available
try:
    import kvikio

    kvikio_available = True
except ImportError:
    kvikio_available = False

# Skip all tests if requirements not met
pytestmark = [
    pytest.mark.skipif(not cuda_available, reason="CUDA not available"),
    pytest.mark.skipif(not kvikio_available, reason="kvikio not available"),
]


# Only import these if we can actually run the tests
if cuda_available and kvikio_available:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.kv_offload.mediums import GPULoadStoreSpec, SSDLoadStoreSpec
    from vllm.v1.kv_offload.worker.gpu_ssd import GpuSsdOffloadingHandler

    BACKENDS_TO_TEST = [FlashAttentionBackend]

    if not current_platform.is_rocm():
        try:
            from vllm.v1.attention.backends.flashinfer import FlashInferBackend

            BACKENDS_TO_TEST.append(FlashInferBackend)
        except ImportError:
            pass

# Test parameters
NUM_GPU_BLOCKS = [64]
NUM_SSD_BLOCKS = [256]
GPU_BLOCK_SIZES = [16]
GPU_BLOCKS_PER_SSD_BLOCK = [1, 2]
HEAD_SIZES = [64]
NUM_HEADS = [8]
NUM_LAYERS = [2]
DTYPES = [torch.bfloat16]
SEEDS = [0]
CUDA_DEVICES = ["cuda:0"]
NUM_MAPPINGS = [3]


@pytest.mark.parametrize("gpu_to_ssd", [True, False])
@pytest.mark.parametrize("num_mappings", NUM_MAPPINGS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("gpu_block_size", GPU_BLOCK_SIZES)
@pytest.mark.parametrize("gpu_blocks_per_ssd_block", GPU_BLOCKS_PER_SSD_BLOCK)
@pytest.mark.parametrize("num_gpu_blocks", NUM_GPU_BLOCKS)
@pytest.mark.parametrize("num_ssd_blocks", NUM_SSD_BLOCKS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_transfer(
    gpu_to_ssd: bool,
    num_mappings: int,
    head_size: int,
    num_heads: int,
    gpu_block_size: int,
    gpu_blocks_per_ssd_block: int,
    num_gpu_blocks: int,
    num_ssd_blocks: int,
    num_layers: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """Test GPU-SSD transfer in both directions."""
    current_platform.seed_everything(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create per-layer GPU KV caches based on available attn_backends
        attn_backends_list = BACKENDS_TO_TEST

        gpu_caches = {}
        attn_backends = {}
        for i in range(num_layers):
            layer_name = f"layer {i}"

            attn_backend = attn_backends_list[i % len(attn_backends_list)]
            attn_backends[layer_name] = attn_backend

            gpu_cache_shape = attn_backend.get_kv_cache_shape(
                num_gpu_blocks, gpu_block_size, num_heads, head_size
            )
            gpu_caches[layer_name] = torch.rand(
                gpu_cache_shape, dtype=dtype, device=device
            )

        # Create handler
        ssd_block_size = gpu_blocks_per_ssd_block * gpu_block_size
        handler = GpuSsdOffloadingHandler(
            attn_backends=attn_backends,
            gpu_block_size=gpu_block_size,
            ssd_block_size=ssd_block_size,
            num_ssd_blocks=num_ssd_blocks,
            gpu_caches=gpu_caches,
            ssd_cache_dir=tmpdir,
        )

        # Select block mappings
        gpu_blocks = random.sample(
            range(num_gpu_blocks), num_mappings * gpu_blocks_per_ssd_block
        )
        ssd_blocks = random.sample(range(num_ssd_blocks), num_mappings)

        # Convert ssd blocks to gpu block size
        ssd_blocks_in_gpu_block_size = []
        for ssd_block in ssd_blocks:
            base_block_id = ssd_block * gpu_blocks_per_ssd_block
            for i in range(gpu_blocks_per_ssd_block):
                ssd_blocks_in_gpu_block_size.append(i + base_block_id)

        # Maybe skip a GPU block to test reading from the middle of an SSD block
        if not gpu_to_ssd:
            gpu_blocks = gpu_blocks[gpu_blocks_per_ssd_block - 1 :]
            ssd_blocks_in_gpu_block_size = ssd_blocks_in_gpu_block_size[
                gpu_blocks_per_ssd_block - 1 :
            ]

        # Set transfer direction
        if gpu_to_ssd:
            src_spec_class = GPULoadStoreSpec
            dst_spec_class = SSDLoadStoreSpec
            src_blocks = gpu_blocks
            dst_blocks = ssd_blocks
        else:
            src_spec_class = SSDLoadStoreSpec
            dst_spec_class = GPULoadStoreSpec
            src_blocks = ssd_blocks
            dst_blocks = gpu_blocks

        # Build transfer specs
        src_spec = src_spec_class(src_blocks)
        dst_spec = dst_spec_class(dst_blocks)

        # Clone GPU tensors before transfer
        orig_gpu_caches = [x.clone() for x in handler.gpu_tensors]

        # Call transfer function
        assert handler.transfer_async(1, (src_spec, dst_spec))
        assert 1 in handler.pending_transfers

        # Wait for transfer to complete
        end_time = time.time() + 30  # Longer timeout for SSD I/O
        while time.time() < end_time:
            finished = handler.get_finished()
            if finished:
                assert finished == [(1, True)]
                break
            time.sleep(0.1)
        else:
            pytest.fail("Transfer did not complete within timeout")

        # Clean up
        handler.close()


@pytest.mark.parametrize("num_layers", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_multiple_transfers(num_layers: int, dtype: torch.dtype) -> None:
    """Test multiple concurrent transfers."""
    current_platform.seed_everything(42)
    device = "cuda:0"

    with tempfile.TemporaryDirectory() as tmpdir:
        attn_backend = BACKENDS_TO_TEST[0]

        gpu_caches = {}
        attn_backends = {}
        for i in range(num_layers):
            layer_name = f"layer {i}"
            attn_backends[layer_name] = attn_backend
            gpu_cache_shape = attn_backend.get_kv_cache_shape(
                num_blocks=64, block_size=16, num_kv_heads=8, head_size=64
            )
            gpu_caches[layer_name] = torch.rand(
                gpu_cache_shape, dtype=dtype, device=device
            )

        handler = GpuSsdOffloadingHandler(
            attn_backends=attn_backends,
            gpu_block_size=16,
            ssd_block_size=16,
            num_ssd_blocks=256,
            gpu_caches=gpu_caches,
            ssd_cache_dir=tmpdir,
            num_io_threads=4,
        )

        # Submit multiple transfers
        num_transfers = 5
        for i in range(num_transfers):
            src_spec = GPULoadStoreSpec([i])
            dst_spec = SSDLoadStoreSpec([i])
            assert handler.transfer_async(i, (src_spec, dst_spec))

        # Wait for all transfers to complete
        completed = set()
        end_time = time.time() + 60
        while len(completed) < num_transfers and time.time() < end_time:
            for job_id, success in handler.get_finished():
                assert success
                completed.add(job_id)
            time.sleep(0.1)

        assert len(completed) == num_transfers

        handler.close()


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_roundtrip_data_integrity(dtype: torch.dtype) -> None:
    """Test that data survives a GPU -> SSD -> GPU roundtrip."""
    current_platform.seed_everything(123)
    device = "cuda:0"

    with tempfile.TemporaryDirectory() as tmpdir:
        attn_backend = BACKENDS_TO_TEST[0]

        gpu_caches = {}
        attn_backends = {}
        layer_name = "layer 0"
        attn_backends[layer_name] = attn_backend
        gpu_cache_shape = attn_backend.get_kv_cache_shape(
            num_blocks=64, block_size=16, num_kv_heads=8, head_size=64
        )
        gpu_caches[layer_name] = torch.rand(
            gpu_cache_shape, dtype=dtype, device=device
        )

        handler = GpuSsdOffloadingHandler(
            attn_backends=attn_backends,
            gpu_block_size=16,
            ssd_block_size=16,
            num_ssd_blocks=256,
            gpu_caches=gpu_caches,
            ssd_cache_dir=tmpdir,
        )

        # Save original data
        original_data = handler.gpu_tensors[0][0].clone()

        # Transfer GPU block 0 to SSD block 0
        src_spec = GPULoadStoreSpec([0])
        dst_spec = SSDLoadStoreSpec([0])
        assert handler.transfer_async(1, (src_spec, dst_spec))

        # Wait for completion
        end_time = time.time() + 30
        while time.time() < end_time:
            finished = handler.get_finished()
            if finished:
                break
            time.sleep(0.1)

        # Modify GPU block 0 to verify roundtrip works
        handler.gpu_tensors[0][0].zero_()

        # Transfer SSD block 0 back to GPU block 0
        src_spec = SSDLoadStoreSpec([0])
        dst_spec = GPULoadStoreSpec([0])
        assert handler.transfer_async(2, (src_spec, dst_spec))

        # Wait for completion
        end_time = time.time() + 30
        while time.time() < end_time:
            finished = handler.get_finished()
            if finished:
                break
            time.sleep(0.1)

        # Verify data matches original
        torch.testing.assert_close(
            handler.gpu_tensors[0][0].cpu(), original_data.cpu()
        )

        handler.close()


class TestSSDLoadStoreSpec:
    """Unit tests for SSDLoadStoreSpec."""

    def test_medium(self):
        """Test medium name."""
        assert SSDLoadStoreSpec.medium() == "SSD"

    def test_block_ids_numpy(self):
        """Test block IDs are stored as numpy array."""
        import numpy as np

        spec = SSDLoadStoreSpec([1, 2, 3])
        assert isinstance(spec.block_ids, np.ndarray)
        assert spec.block_ids.dtype == np.int64
        assert list(spec.block_ids) == [1, 2, 3]
