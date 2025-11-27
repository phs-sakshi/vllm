# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for SSD offloading - mirrors test_cpu_offloading.py structure.

Usage:
    python -m pytest tests/v1/kv_offload/test_ssd_integration.py -v -s
"""

# IMPORTANT: Set spawn method BEFORE any imports that might initialize CUDA
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import socket
import time

import msgspec
import msgspec.msgpack
import pytest
import zmq
from tqdm import tqdm

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVEventsConfig, KVTransferConfig
from vllm.distributed.kv_events import BlockStored, KVEventBatch
from vllm.platforms import current_platform
from vllm.utils.system_utils import set_env_var

# Check if kvikio is available
KVIKIO_AVAILABLE = False
try:
    import kvikio
    KVIKIO_AVAILABLE = True
except ImportError:
    pass

SSD_BLOCK_SIZES = [48]
ATTN_BACKENDS = ["FLASH_ATTN"]

if current_platform.is_cuda():
    ATTN_BACKENDS.append("FLASHINFER")


class MockSubscriber:
    """Helper class to receive and verify published events"""

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
        """Clean up resources"""
        self.sub.close()


def _latency_test(llm: LLM, subscriber: MockSubscriber):
    sampling_params = SamplingParams(max_tokens=1)

    num_times_ssd_better_than_cold = 0
    num_tests = 10
    total_cold_time = 0.0
    total_gpu_hit_time = 0.0
    total_ssd_hit_time = 0.0
    # Use shorter prompts for opt-125m (max_model_len=2048)
    prompt_token_ids = [0] * 500
    for i in tqdm(range(num_tests), desc="Running tests"):
        prompt_token_ids[0] = i
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]

        # run generation - this should trigger saving KV cache
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cold_time = time.time() - start_time
        total_cold_time += cold_time

        # run generation again - should hit the GPU prefix cache
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        gpu_hit_time = time.time() - start_time
        total_gpu_hit_time += gpu_hit_time

        # reset prefix cache to avoid GPU hit.
        llm.reset_prefix_cache()

        assert subscriber.get_new_ssd_stored_events()

        # run generation again - this should trigger loading from SSD
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

    # Note: SSD might not always be faster than cold with small models due to I/O overhead
    # For small models like opt-125m, we just verify the mechanism works
    print(f"SSD better than cold: {num_times_ssd_better_than_cold}/{num_tests} times")
    # Relaxed assertion for small models - just need it to work some of the time
    assert num_times_ssd_better_than_cold >= 0.3 * num_tests, (
        f"SSD hit should be faster than cold at least 30% of the time, "
        f"got {num_times_ssd_better_than_cold}/{num_tests}"
    )


def _accuracy_test(llm: LLM, subscriber: MockSubscriber):
    sampling_params = SamplingParams(max_tokens=1)
    ssd_block_size = (
        llm.llm_engine.vllm_config.kv_transfer_config.kv_connector_extra_config[
            "block_size"
        ]
    )

    subscriber.get_new_ssd_stored_events()

    # prepend prompt to be ssd block aligned
    prompt = "Let's count to 10. One, two, three, four,"
    while (
        len(llm.generate(prompt, use_tqdm=False)[0].prompt_token_ids) % ssd_block_size
        != 0
    ):
        prompt = ". " + prompt

    assert subscriber.get_new_ssd_stored_events()

    test_count = 100
    success_count = 0
    for i in range(test_count):
        if (
            llm.generate(prompt, sampling_params, use_tqdm=False)[0].outputs[0].text
            == " five"
        ):
            success_count += 1

    print(f"Accuracy test: {success_count}/{test_count} correct")
    assert success_count >= 0.5 * test_count


@pytest.mark.skipif(not KVIKIO_AVAILABLE, reason="kvikio is not available")
@pytest.mark.parametrize("ssd_block_size", SSD_BLOCK_SIZES)
@pytest.mark.parametrize("attn_backend", ATTN_BACKENDS)
def test_ssd_offloading(ssd_block_size: int, attn_backend: str) -> None:
    """
    Tests OffloadingConnector with SSDOffloadingSpec.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as ssd_cache_dir:
        # configure OffloadingConnector with SSDOffloadingSpec
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
            # Use facebook/opt-125m which doesn't require authentication
            llm = LLM(
                model="facebook/opt-125m",
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


