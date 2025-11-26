# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for SSD backend and manager functionality.

These tests verify the SSDBackend class works correctly with
LRU and ARC managers, similar to the CPU backend tests.
"""
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.ssd import SSDBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import SSDLoadStoreSpec


@dataclass
class ExpectedPrepareStoreOutput:
    block_hashes_to_store: list[int]
    store_block_ids: list[int]
    block_hashes_evicted: list[int]


def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
    return [BlockHash(str(i).encode()) for i in int_hashes]


def verify_store_output(
    prepare_store_output: PrepareStoreOutput | None,
    expected_prepare_store_output: ExpectedPrepareStoreOutput,
):
    assert prepare_store_output is not None
    assert prepare_store_output.block_hashes_to_store == to_hashes(
        expected_prepare_store_output.block_hashes_to_store
    )
    assert prepare_store_output.block_hashes_evicted == to_hashes(
        expected_prepare_store_output.block_hashes_evicted
    )
    store_spec = prepare_store_output.store_spec
    assert isinstance(store_spec, SSDLoadStoreSpec)
    expected_array = np.array(
        expected_prepare_store_output.store_block_ids, dtype=np.int64
    )
    assert np.array_equal(expected_array, store_spec.block_ids)


def verify_load_output(
    prepare_load_output: LoadStoreSpec, expected_prepare_load_output: list[int]
):
    assert isinstance(prepare_load_output, SSDLoadStoreSpec)
    expected_array = np.array(expected_prepare_load_output, dtype=np.int64)
    assert np.array_equal(expected_array, prepare_load_output.block_ids)


def verify_events(
    events: Iterable[OffloadingEvent],
    block_size: int,
    expected_stores: tuple[set[int], ...] = (),
    expected_evictions: tuple[set[int], ...] = (),
):
    stores: list[set[BlockHash]] = []
    evictions: list[set[BlockHash]] = []
    for event in events:
        assert event.medium == SSDLoadStoreSpec.medium()
        assert event.block_size == block_size
        if event.removed:
            evictions.append(set(event.block_hashes))
        else:
            stores.append(set(event.block_hashes))

    def to_hash_sets(int_sets: tuple[set[int], ...]) -> tuple[set[BlockHash], ...]:
        return tuple([set(to_hashes(list(int_set))) for int_set in int_sets])

    assert tuple(evictions) == to_hash_sets(expected_evictions)
    assert tuple(stores) == to_hash_sets(expected_stores)


class TestSSDBackend:
    """Tests for SSDBackend class."""

    def test_backend_initialization(self):
        """Test that SSD backend initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            block_size = 256
            num_blocks = 4
            ssd_backend = SSDBackend(
                block_size=block_size,
                num_blocks=num_blocks,
                ssd_cache_dir=tmpdir,
            )

            assert ssd_backend.block_size == block_size
            assert ssd_backend.num_blocks == num_blocks
            assert ssd_backend.medium == "SSD"
            assert ssd_backend.get_num_free_blocks() == num_blocks
            assert Path(tmpdir).exists()

    def test_allocate_and_free_blocks(self):
        """Test block allocation and freeing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ssd_backend = SSDBackend(
                block_size=256,
                num_blocks=4,
                ssd_cache_dir=tmpdir,
            )

            # Initially all blocks are free
            assert ssd_backend.get_num_free_blocks() == 4

            # Allocate 2 blocks
            blocks = ssd_backend.allocate_blocks(to_hashes([1, 2]))
            assert len(blocks) == 2
            assert ssd_backend.get_num_free_blocks() == 2

            # Allocate 2 more blocks
            blocks2 = ssd_backend.allocate_blocks(to_hashes([3, 4]))
            assert len(blocks2) == 2
            assert ssd_backend.get_num_free_blocks() == 0

            # Free one block
            ssd_backend.free(blocks[0])
            assert ssd_backend.get_num_free_blocks() == 1

            # Allocate using freed block
            blocks3 = ssd_backend.allocate_blocks(to_hashes([5]))
            assert len(blocks3) == 1
            assert ssd_backend.get_num_free_blocks() == 0

    def test_get_load_store_spec(self):
        """Test generating load/store specs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ssd_backend = SSDBackend(
                block_size=256,
                num_blocks=4,
                ssd_cache_dir=tmpdir,
            )

            blocks = ssd_backend.allocate_blocks(to_hashes([1, 2, 3]))
            spec = ssd_backend.get_load_store_spec(to_hashes([1, 2, 3]), blocks)

            assert isinstance(spec, SSDLoadStoreSpec)
            assert len(spec.block_ids) == 3
            assert list(spec.block_ids) == [0, 1, 2]

    def test_cache_file_path(self):
        """Test cache file path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ssd_backend = SSDBackend(
                block_size=256,
                num_blocks=4,
                ssd_cache_dir=tmpdir,
                cache_file_prefix="test_cache",
            )

            path0 = ssd_backend.get_cache_file_path(0)
            assert path0 == Path(tmpdir) / "test_cache_layer_0.bin"

            path1 = ssd_backend.get_cache_file_path(1)
            assert path1 == Path(tmpdir) / "test_cache_layer_1.bin"

    def test_cleanup(self):
        """Test cache file cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ssd_backend = SSDBackend(
                block_size=256,
                num_blocks=4,
                ssd_cache_dir=tmpdir,
                cache_file_prefix="test_cleanup",
            )

            # Create some cache files
            cache_file = Path(tmpdir) / "test_cleanup_layer_0.bin"
            cache_file.write_bytes(b"\x00" * 1024)

            # Cleanup should remove the file
            ssd_backend.cleanup()
            assert not cache_file.exists()


class TestSSDManagerLRU:
    """Tests for LRU manager with SSD backend."""

    def test_ssd_manager_basic(self):
        """
        Tests LRUOffloadingManager with an SSDBackend.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # initialize an SSD backend with a capacity of 4 blocks
            block_size = 256
            ssd_backend = SSDBackend(
                block_size=block_size, num_blocks=4, ssd_cache_dir=tmpdir
            )
            ssd_manager = LRUOffloadingManager(ssd_backend, enable_events=True)

            # prepare store [1, 2]
            prepare_store_output = ssd_manager.prepare_store(to_hashes([1, 2]))
            verify_store_output(
                prepare_store_output,
                ExpectedPrepareStoreOutput(
                    block_hashes_to_store=[1, 2],
                    store_block_ids=[0, 1],
                    block_hashes_evicted=[],
                ),
            )

            # lookup [1, 2] -> not ready
            assert ssd_manager.lookup(to_hashes([1, 2])) == 0

            # no events so far
            assert list(ssd_manager.take_events()) == []

            # complete store [1, 2]
            ssd_manager.complete_store(to_hashes([1, 2]))
            verify_events(
                ssd_manager.take_events(),
                block_size=block_size,
                expected_stores=({1, 2},),
            )

            # lookup [1, 2]
            assert ssd_manager.lookup(to_hashes([1])) == 1
            assert ssd_manager.lookup(to_hashes([1, 2])) == 2
            assert ssd_manager.lookup(to_hashes([1, 2, 3])) == 2

    def test_ssd_manager_eviction(self):
        """
        Tests eviction behavior with SSD backend.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            block_size = 256
            ssd_backend = SSDBackend(
                block_size=block_size, num_blocks=4, ssd_cache_dir=tmpdir
            )
            ssd_manager = LRUOffloadingManager(ssd_backend, enable_events=True)

            # fill cache with [1, 2, 3, 4]
            prepare_store_output = ssd_manager.prepare_store(to_hashes([1, 2, 3, 4]))
            verify_store_output(
                prepare_store_output,
                ExpectedPrepareStoreOutput(
                    block_hashes_to_store=[1, 2, 3, 4],
                    store_block_ids=[0, 1, 2, 3],
                    block_hashes_evicted=[],
                ),
            )
            ssd_manager.complete_store(to_hashes([1, 2, 3, 4]))

            # prepare store [5] -> evicts [1]
            prepare_store_output = ssd_manager.prepare_store(to_hashes([5]))
            verify_store_output(
                prepare_store_output,
                ExpectedPrepareStoreOutput(
                    block_hashes_to_store=[5],
                    store_block_ids=[0],
                    block_hashes_evicted=[1],
                ),
            )

    def test_ssd_manager_load(self):
        """
        Tests loading blocks from SSD.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            block_size = 256
            ssd_backend = SSDBackend(
                block_size=block_size, num_blocks=4, ssd_cache_dir=tmpdir
            )
            ssd_manager = LRUOffloadingManager(ssd_backend, enable_events=True)

            # store [1, 2, 3, 4]
            ssd_manager.prepare_store(to_hashes([1, 2, 3, 4]))
            ssd_manager.complete_store(to_hashes([1, 2, 3, 4]))

            # prepare load [2, 3]
            prepare_load_output = ssd_manager.prepare_load(to_hashes([2, 3]))
            verify_load_output(prepare_load_output, [1, 2])

            # complete load
            ssd_manager.complete_load(to_hashes([2, 3]))

            # blocks should still be available
            assert ssd_manager.lookup(to_hashes([2, 3])) == 2

    def test_ssd_manager_touch(self):
        """
        Tests touch operation for LRU ordering.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            block_size = 256
            ssd_backend = SSDBackend(
                block_size=block_size, num_blocks=4, ssd_cache_dir=tmpdir
            )
            ssd_manager = LRUOffloadingManager(ssd_backend, enable_events=True)

            # store [1, 2, 3, 4]
            ssd_manager.prepare_store(to_hashes([1, 2, 3, 4]))
            ssd_manager.complete_store(to_hashes([1, 2, 3, 4]))

            # touch [1, 2] to move them to end of LRU
            ssd_manager.touch(to_hashes([1, 2]))

            # store [5, 6] -> should evict [3, 4] (oldest after touch)
            prepare_store_output = ssd_manager.prepare_store(to_hashes([5, 6]))
            verify_store_output(
                prepare_store_output,
                ExpectedPrepareStoreOutput(
                    block_hashes_to_store=[5, 6],
                    store_block_ids=[2, 3],
                    block_hashes_evicted=[3, 4],
                ),
            )


class TestSSDManagerARC:
    """Tests for ARC manager with SSD backend."""

    def test_arc_manager_basic(self):
        """
        Tests ARCOffloadingManager basic operations with an SSDBackend.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            block_size = 256
            ssd_backend = SSDBackend(
                block_size=block_size, num_blocks=4, ssd_cache_dir=tmpdir
            )
            arc_manager = ARCOffloadingManager(ssd_backend, enable_events=True)

            # prepare store [1, 2]
            prepare_store_output = arc_manager.prepare_store(to_hashes([1, 2]))
            verify_store_output(
                prepare_store_output,
                ExpectedPrepareStoreOutput(
                    block_hashes_to_store=[1, 2],
                    store_block_ids=[0, 1],
                    block_hashes_evicted=[],
                ),
            )

            # lookup [1, 2] -> not ready
            assert arc_manager.lookup(to_hashes([1, 2])) == 0

            # complete store [1, 2]
            arc_manager.complete_store(to_hashes([1, 2]))

            # lookup [1, 2]
            assert arc_manager.lookup(to_hashes([1])) == 1
            assert arc_manager.lookup(to_hashes([1, 2])) == 2

            # blocks should be in T1 (recent)
            assert len(arc_manager.t1) == 2
            assert len(arc_manager.t2) == 0

    def test_arc_manager_t1_to_t2_promotion(self):
        """
        Tests that accessing a block in T1 promotes it to T2.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            block_size = 256
            ssd_backend = SSDBackend(
                block_size=block_size, num_blocks=4, ssd_cache_dir=tmpdir
            )
            arc_manager = ARCOffloadingManager(ssd_backend, enable_events=False)

            # store and complete block 1
            arc_manager.prepare_store(to_hashes([1]))
            arc_manager.complete_store(to_hashes([1]))

            # block 1 starts in T1
            assert to_hashes([1])[0] in arc_manager.t1
            assert to_hashes([1])[0] not in arc_manager.t2

            # touch block 1 (simulate second access)
            arc_manager.touch(to_hashes([1]))

            # block 1 should now be in T2 (frequent)
            assert to_hashes([1])[0] not in arc_manager.t1
            assert to_hashes([1])[0] in arc_manager.t2

    def test_arc_manager_eviction(self):
        """
        Tests ARC eviction behavior with SSD backend.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            block_size = 256
            ssd_backend = SSDBackend(
                block_size=block_size, num_blocks=4, ssd_cache_dir=tmpdir
            )
            arc_manager = ARCOffloadingManager(ssd_backend, enable_events=True)

            # store [1, 2, 3, 4]
            prepare_store_output = arc_manager.prepare_store(to_hashes([1, 2, 3, 4]))
            verify_store_output(
                prepare_store_output,
                ExpectedPrepareStoreOutput(
                    block_hashes_to_store=[1, 2, 3, 4],
                    store_block_ids=[0, 1, 2, 3],
                    block_hashes_evicted=[],
                ),
            )
            arc_manager.complete_store(to_hashes([1, 2, 3, 4]))

            # prepare load [2, 3] (increases ref_cnt)
            prepare_load_output = arc_manager.prepare_load(to_hashes([2, 3]))
            verify_load_output(prepare_load_output, [1, 2])

            # prepare store [5, 6, 7] should fail (blocks being loaded)
            assert arc_manager.prepare_store(to_hashes([5, 6, 7])) is None

            # complete load
            arc_manager.complete_load(to_hashes([2, 3]))

            # now store should succeed
            prepare_store_output = arc_manager.prepare_store(to_hashes([5, 6, 7]))
            assert prepare_store_output is not None
            assert len(prepare_store_output.block_hashes_evicted) >= 1


class TestSSDLoadStoreSpec:
    """Tests for SSDLoadStoreSpec class."""

    def test_medium_name(self):
        """Test that SSD medium name is correct."""
        assert SSDLoadStoreSpec.medium() == "SSD"

    def test_block_ids(self):
        """Test block IDs storage."""
        spec = SSDLoadStoreSpec([1, 2, 3, 4, 5])
        assert len(spec.block_ids) == 5
        assert list(spec.block_ids) == [1, 2, 3, 4, 5]

    def test_repr(self):
        """Test string representation."""
        spec = SSDLoadStoreSpec([1, 2])
        assert "1" in repr(spec)
        assert "2" in repr(spec)
