# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for SSD KV cache offloading with GPU Direct Storage.

These tests verify the complete SSD offloading pipeline works correctly
with the OffloadingConnector and SSDOffloadingSpec.
"""
import socket
import tempfile
import time

import msgspec
import msgspec.msgpack
import pytest
import torch
import zmq
from tqdm import tqdm

from vllm.platforms import current_platform
from vllm.utils.system_utils import set_env_var

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

# Only import vllm modules if we can run the tests
if cuda_available and kvikio_available:
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.config import KVEventsConfig, KVTransferConfig
    from vllm.distributed.kv_events import BlockStored, KVEventBatch

SSD_BLOCK_SIZES = [48]
ATTN_BACKENDS = ["FLASH_ATTN"]

if cuda_available and current_platform.is_cuda():
    ATTN_BACKENDS.append("FLASHINFER")


class MockSubscriber:
    """Helper class to receive and verify published events."""

    def __init__(
        self,
        endpoint: str,
        topic: str,
    ):
        self.ctx = zmq.Context.instance()
        self.topic_bytes = topic.encode("utf-8")

        # Set up subscriber socket
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, self.topic_bytes)
        self.sub.connect(endpoint)

        self.decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

    def get_new_ssd_stored_events(self) -> list[BlockStored]:
        """Get new SSD stored events."""
        ssd_stored_events: list[BlockStored] = []

        poller = zmq.Poller()
        poller.register(self.sub, zmq.POLLIN)

        timeout = 1000  # 1 second
        while True:
            events = dict(poller.poll(timeout))

            if events.get(self.sub) != zmq.POLLIN:
                return ssd_stored_events

            topic_bytes, _, payload = self.sub.recv_multipart()

            assert topic_bytes == self.topic_bytes

            event_batch = self.decoder.decode(payload)
            assert isinstance(event_batch, KVEventBatch)
            for event in event_batch.events:
                if isinstance(event, BlockStored) and event.medium == "SSD":
                    ssd_stored_events.append(event)
                    timeout = 100

    def close(self):
        """Clean up resources."""
        self.sub.close()


def _latency_test(llm: LLM, subscriber: MockSubscriber):
    """Test that SSD offloading improves latency vs cold start."""
    sampling_params = SamplingParams(max_tokens=1)

    num_times_ssd_better_than_cold = 0
    num_tests = 5  # Fewer tests for SSD due to slower I/O
    total_cold_time = 0.0
    total_gpu_hit_time = 0.0
    total_ssd_hit_time = 0.0
    prompt_token_ids = [0] * 5001  # Shorter prompts for faster testing
    for i in tqdm(range(num_tests), desc="Running latency tests"):
        prompt_token_ids[0] = i
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]

        # Run generation - this should trigger saving KV cache
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cold_time = time.time() - start_time
        total_cold_time += cold_time

        # Run generation again - should hit the GPU prefix cache
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        gpu_hit_time = time.time() - start_time
        total_gpu_hit_time += gpu_hit_time

        # Reset prefix cache to avoid GPU hit
        llm.reset_prefix_cache()

        # Wait for SSD store events
        ssd_events = subscriber.get_new_ssd_stored_events()
        assert ssd_events, "Expected SSD store events"

        # Run generation again - this should trigger loading from SSD
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        ssd_hit_time = time.time() - start_time
        total_ssd_hit_time += ssd_hit_time

        if ssd_hit_time < cold_time:
            num_times_ssd_better_than_cold += 1

    print("Average times:")
    print(f"    Cold: {total_cold_time * 1000 / num_tests:.2f}ms")
    print(f"    GPU hit: {total_gpu_hit_time * 1000 / num_tests:.2f}ms")
    print(f"    SSD hit: {total_ssd_hit_time * 1000 / num_tests:.2f}ms")

    # SSD should be faster than cold start most of the time
    # (lower threshold than CPU due to SSD latency variance)
    assert num_times_ssd_better_than_cold >= 0.6 * num_tests


def _accuracy_test(llm: LLM, subscriber: MockSubscriber):
    """Test that SSD offloading produces correct results."""
    sampling_params = SamplingParams(max_tokens=1)
    ssd_block_size = (
        llm.llm_engine.vllm_config.kv_transfer_config.kv_connector_extra_config[
            "block_size"
        ]
    )

    subscriber.get_new_ssd_stored_events()

    # Prepend prompt to be SSD block aligned
    prompt = "Let's count to 10. One, two, three, four,"
    while (
        len(llm.generate(prompt, use_tqdm=False)[0].prompt_token_ids) % ssd_block_size
        != 0
    ):
        prompt = ". " + prompt

    assert subscriber.get_new_ssd_stored_events()

    test_count = 50  # Fewer tests for SSD
    success_count = 0
    for i in range(test_count):
        if (
            llm.generate(prompt, sampling_params, use_tqdm=False)[0].outputs[0].text
            == " five"
        ):
            success_count += 1

    assert success_count >= 0.5 * test_count


@pytest.mark.parametrize("ssd_block_size", SSD_BLOCK_SIZES)
@pytest.mark.parametrize("attn_backend", ATTN_BACKENDS)
def test_ssd_offloading(ssd_block_size: int, attn_backend: str) -> None:
    """
    Tests OffloadingConnector with SSDOffloadingSpec.
    """
    with tempfile.TemporaryDirectory() as ssd_cache_dir:
        # Configure OffloadingConnector with SSDOffloadingSpec
        kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "SSDOffloadingSpec",
                "num_ssd_blocks": 1000,
                "block_size": ssd_block_size,
                "ssd_cache_dir": ssd_cache_dir,
                "eviction_policy": "lru",
                "num_io_threads": 4,
            },
        )

        port: int
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

        with set_env_var("VLLM_ATTENTION_BACKEND", attn_backend):
            llm = LLM(
                model="meta-llama/Llama-3.2-1B-Instruct",
                gpu_memory_utilization=0.5,
                kv_events_config=kv_events_config,
                kv_transfer_config=kv_transfer_config,
            )

        events_endpoint = events_endpoint.replace("*", "127.0.0.1")
        subscriber = MockSubscriber(events_endpoint, topic=kv_events_config.topic)

        try:
            _latency_test(llm, subscriber)
            _accuracy_test(llm, subscriber)
        finally:
            subscriber.close()
            del llm


@pytest.mark.parametrize("eviction_policy", ["lru", "arc"])
def test_ssd_offloading_eviction_policies(eviction_policy: str) -> None:
    """
    Tests SSD offloading with different eviction policies.
    """
    with tempfile.TemporaryDirectory() as ssd_cache_dir:
        kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "SSDOffloadingSpec",
                "num_ssd_blocks": 500,
                "block_size": 32,
                "ssd_cache_dir": ssd_cache_dir,
                "eviction_policy": eviction_policy,
            },
        )

        with set_env_var("VLLM_ATTENTION_BACKEND", "FLASH_ATTN"):
            llm = LLM(
                model="meta-llama/Llama-3.2-1B-Instruct",
                gpu_memory_utilization=0.5,
                kv_transfer_config=kv_transfer_config,
            )

        sampling_params = SamplingParams(max_tokens=10)
        prompt = "Hello, how are you doing today?"

        # Run a few generations to trigger offloading
        for _ in range(3):
            output = llm.generate(prompt, sampling_params, use_tqdm=False)
            assert len(output) > 0
            assert len(output[0].outputs[0].text) > 0

        del llm


def test_ssd_offloading_spec_validation() -> None:
    """
    Tests that SSDOffloadingSpec validates configuration correctly.
    """
    with tempfile.TemporaryDirectory() as ssd_cache_dir:
        # Test missing num_ssd_blocks
        with pytest.raises(ValueError, match="num_ssd_blocks"):
            kv_transfer_config = KVTransferConfig(
                kv_connector="OffloadingConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "spec_name": "SSDOffloadingSpec",
                    "ssd_cache_dir": ssd_cache_dir,
                },
            )
            with set_env_var("VLLM_ATTENTION_BACKEND", "FLASH_ATTN"):
                llm = LLM(
                    model="meta-llama/Llama-3.2-1B-Instruct",
                    gpu_memory_utilization=0.5,
                    kv_transfer_config=kv_transfer_config,
                )


@pytest.mark.parametrize("num_io_threads", [1, 4, 8])
def test_ssd_offloading_io_threads(num_io_threads: int) -> None:
    """
    Tests SSD offloading with different numbers of I/O threads.
    """
    with tempfile.TemporaryDirectory() as ssd_cache_dir:
        kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "SSDOffloadingSpec",
                "num_ssd_blocks": 500,
                "block_size": 32,
                "ssd_cache_dir": ssd_cache_dir,
                "num_io_threads": num_io_threads,
            },
        )

        with set_env_var("VLLM_ATTENTION_BACKEND", "FLASH_ATTN"):
            llm = LLM(
                model="meta-llama/Llama-3.2-1B-Instruct",
                gpu_memory_utilization=0.5,
                kv_transfer_config=kv_transfer_config,
            )

        sampling_params = SamplingParams(max_tokens=5)
        output = llm.generate("Hello world", sampling_params, use_tqdm=False)
        assert len(output) > 0

        del llm
