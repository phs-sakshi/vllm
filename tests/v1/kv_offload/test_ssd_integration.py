# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for SSD offloading using a non-gated model.
"""

import multiprocessing
import os
import socket
import tempfile

import pytest

# Check if kvikio is available (without touching CUDA yet)
KVIKIO_AVAILABLE = False
try:
    import kvikio
    KVIKIO_AVAILABLE = True
except ImportError:
    pass


def _check_cuda_available():
    """Check CUDA availability without initializing it."""
    import torch
    return torch.cuda.is_available()


skip_if_no_cuda = pytest.mark.skipif(
    not _check_cuda_available(),
    reason="CUDA is not available"
)

skip_if_no_kvikio = pytest.mark.skipif(
    not KVIKIO_AVAILABLE,
    reason="kvikio is not available"
)


@skip_if_no_cuda
@skip_if_no_kvikio
def test_ssd_offloading_opt():
    """
    Tests SSD offloading with facebook/opt-125m (no authentication required).
    """
    import torch
    from vllm import LLM
    from vllm.config import KVEventsConfig, KVTransferConfig

    with tempfile.TemporaryDirectory() as ssd_cache_dir:
        # Configure OffloadingConnector with SSDOffloadingSpec
        kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "SSDOffloadingSpec",
                "num_ssd_blocks": 500,
                "block_size": 16,  # Match GPU block size
                "ssd_cache_dir": ssd_cache_dir,
                "eviction_policy": "lru",
                "num_io_threads": 4,
            },
        )

        # Find a free port for events
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            port = s.getsockname()[1]

        events_endpoint = f"tcp://*:{port}"
        kv_events_config = KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=events_endpoint,
            topic="test",
        )

        llm = None
        try:
            # Use facebook/opt-125m which doesn't require authentication
            llm = LLM(
                model="facebook/opt-125m",
                gpu_memory_utilization=0.5,
                kv_events_config=kv_events_config,
                kv_transfer_config=kv_transfer_config,
                max_model_len=512,  # Limit context length for faster test
            )

            # Run a simple generation
            prompts = [
                "Hello, my name is",
                "The quick brown fox",
                "Once upon a time",
            ]
            
            outputs = llm.generate(prompts, sampling_params={"max_tokens": 20})
            
            # Verify we got outputs
            assert len(outputs) == len(prompts)
            for output in outputs:
                assert len(output.outputs) > 0
                assert len(output.outputs[0].text) > 0
                print(f"Prompt: {output.prompt!r}")
                print(f"Output: {output.outputs[0].text!r}")
                print()

            print("SSD offloading integration test PASSED!")

        finally:
            # Cleanup
            if llm is not None:
                del llm
            torch.cuda.empty_cache()


@skip_if_no_cuda
@skip_if_no_kvikio
def test_ssd_offloading_repeated_prompts():
    """
    Tests SSD offloading with repeated prompts to exercise cache hits.
    """
    import torch
    from vllm import LLM
    from vllm.config import KVEventsConfig, KVTransferConfig

    with tempfile.TemporaryDirectory() as ssd_cache_dir:
        kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "SSDOffloadingSpec",
                "num_ssd_blocks": 200,
                "block_size": 16,
                "ssd_cache_dir": ssd_cache_dir,
                "eviction_policy": "lru",
                "num_io_threads": 2,
            },
        )

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            port = s.getsockname()[1]

        kv_events_config = KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=f"tcp://*:{port}",
            topic="test",
        )

        llm = None
        try:
            llm = LLM(
                model="facebook/opt-125m",
                gpu_memory_utilization=0.4,  # Lower to force more offloading
                kv_events_config=kv_events_config,
                kv_transfer_config=kv_transfer_config,
                max_model_len=256,
            )

            # Same prompt multiple times to test cache behavior
            prompt = "The meaning of life is"
            
            # First run
            outputs1 = llm.generate([prompt], sampling_params={"max_tokens": 10, "seed": 42})
            result1 = outputs1[0].outputs[0].text
            print(f"First run: {result1!r}")

            # Second run with same prompt (should potentially hit cache)
            outputs2 = llm.generate([prompt], sampling_params={"max_tokens": 10, "seed": 42})
            result2 = outputs2[0].outputs[0].text
            print(f"Second run: {result2!r}")

            # Results should be the same with same seed
            assert result1 == result2, f"Results differ: {result1!r} vs {result2!r}"
            
            print("Repeated prompts test PASSED!")

        finally:
            if llm is not None:
                del llm
            torch.cuda.empty_cache()


def _run_test_in_subprocess(test_func):
    """Run a test function in a subprocess to avoid CUDA initialization issues."""
    import sys
    
    def wrapper():
        # Set environment variables to help with CUDA
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        test_func()
    
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=wrapper)
    p.start()
    p.join()
    
    if p.exitcode != 0:
        raise RuntimeError(f"Test failed with exit code {p.exitcode}")


if __name__ == "__main__":
    # Must set spawn method BEFORE any CUDA operations
    # This is required because vLLM uses multiprocessing internally
    multiprocessing.set_start_method("spawn", force=True)
    
    print("Running test_ssd_offloading_opt...")
    _run_test_in_subprocess(test_ssd_offloading_opt)
    
    print("\nRunning test_ssd_offloading_repeated_prompts...")
    _run_test_in_subprocess(test_ssd_offloading_repeated_prompts)
    
    print("\nAll integration tests completed!")
