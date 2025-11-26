# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SSD Backend for KV cache offloading using GPU Direct Storage (GDS).

This backend manages block allocation on SSD storage and provides
the necessary specifications for GPU-SSD data transfers.
"""
import ctypes
import os
from collections.abc import Iterable
from pathlib import Path

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.mediums import SSDLoadStoreSpec

logger = init_logger(__name__)


class SSDBlockStatus(BlockStatus):
    """Block status for SSD-backed blocks.

    Contains the block ID which maps to a file offset in the SSD cache file.
    """

    _fields_ = BlockStatus._fields_ + [("block_id", ctypes.c_int64)]  # type: ignore

    def __init__(self, block_id: int):
        super().__init__()
        self.block_id = block_id


class SSDBackend(Backend):
    """Backend for managing KV cache blocks on SSD storage.

    This backend allocates blocks on SSD storage for KV cache offloading.
    Each block corresponds to a fixed-size region in a pre-allocated
    cache file on the SSD.

    The actual I/O operations are performed by the GpuSsdOffloadingHandler
    using GPU Direct Storage (GDS) for efficient GPU-SSD transfers.

    Args:
        block_size: Number of tokens per block (for tracking purposes).
        num_blocks: Maximum number of blocks that can be stored on SSD.
        ssd_cache_dir: Directory path for the SSD cache files.
        cache_file_prefix: Prefix for cache file names (default: "kv_cache").
    """

    def __init__(
        self,
        block_size: int,
        num_blocks: int,
        ssd_cache_dir: str,
        cache_file_prefix: str = "kv_cache",
    ):
        super().__init__(block_size=block_size, medium=SSDLoadStoreSpec.medium())

        self.num_blocks: int = num_blocks
        self.num_allocated_blocks: int = 0
        self.allocated_blocks_free_list: list[int] = []

        # SSD storage configuration
        self.ssd_cache_dir = Path(ssd_cache_dir)
        self.cache_file_prefix = cache_file_prefix

        # Ensure cache directory exists
        self.ssd_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initialized SSD backend with %d blocks in %s",
            num_blocks,
            self.ssd_cache_dir,
        )

    def get_cache_file_path(self, layer_idx: int = 0) -> Path:
        """Get the path to the cache file for a given layer.

        Args:
            layer_idx: Index of the layer (for multi-file setups).

        Returns:
            Path to the cache file.
        """
        return self.ssd_cache_dir / f"{self.cache_file_prefix}_layer_{layer_idx}.bin"

    def get_num_free_blocks(self) -> int:
        """Returns the number of blocks available for allocation."""
        return (
            len(self.allocated_blocks_free_list)
            + self.num_blocks
            - self.num_allocated_blocks
        )

    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        """Allocate SSD blocks for the given block hashes.

        Args:
            block_hashes: List of block hashes to allocate storage for.

        Returns:
            List of SSDBlockStatus objects representing the allocated blocks.
        """
        num_fresh_blocks = min(
            len(block_hashes), self.num_blocks - self.num_allocated_blocks
        )
        num_reused_blocks = len(block_hashes) - num_fresh_blocks
        assert len(self.allocated_blocks_free_list) >= num_reused_blocks

        blocks: list[BlockStatus] = []

        # Allocate fresh blocks (never used before)
        for _ in range(num_fresh_blocks):
            blocks.append(SSDBlockStatus(self.num_allocated_blocks))
            self.num_allocated_blocks += 1

        # Reuse previously freed blocks
        for _ in range(num_reused_blocks):
            block_id = self.allocated_blocks_free_list.pop()
            blocks.append(SSDBlockStatus(block_id))

        return blocks

    def free(self, block: BlockStatus):
        """Free a previously allocated block.

        Args:
            block: The block to be freed.
        """
        assert isinstance(block, SSDBlockStatus)
        self.allocated_blocks_free_list.append(block.block_id)

    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        """Get the specification for loading/storing blocks.

        Args:
            block_hashes: The block hashes identifying the blocks.
            blocks: The blocks to create a spec for.

        Returns:
            SSDLoadStoreSpec containing the block IDs.
        """
        return SSDLoadStoreSpec([block.block_id for block in blocks])

    def cleanup(self):
        """Clean up SSD cache files.

        This method removes all cache files created by this backend.
        Should be called when the backend is no longer needed.
        """
        for cache_file in self.ssd_cache_dir.glob(f"{self.cache_file_prefix}_*.bin"):
            try:
                os.remove(cache_file)
                logger.debug("Removed cache file: %s", cache_file)
            except OSError as e:
                logger.warning("Failed to remove cache file %s: %s", cache_file, e)
