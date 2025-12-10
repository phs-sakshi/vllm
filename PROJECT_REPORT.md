# SSD-Based KV Cache PagedAttention Implementation in vLLM

## Project Report

---

## 1. Problem Statement and Motivation

### 1.1 The Challenge: GPU Memory Constraints in LLM Serving

Large Language Model (LLM) serving systems face a critical bottleneck: **GPU memory scarcity**. During the decoding phase of text generation, the Key-Value (KV) cache dominates memory consumption. For modern models like LLaMA-70B with long context windows (128K+ tokens), a single request can require gigabytes of KV cache storage. This creates severe limitations on:

- **Batch size**: Fewer concurrent requests can be served
- **Context length**: Longer conversations exhaust memory quickly  
- **Reuse opportunities**: Valuable cached computations are discarded prematurely

### 1.2 Current State: PagedAttention and CPU Offloading

vLLM's **PagedAttention** architecture was a breakthrough in addressing memory fragmentation. It stores KV tensors in non-contiguous, fixed-size blocks ("pages"), enabling:

- Fine-grained memory allocation
- Block-level reuse across requests (prefix caching)
- Reduced internal fragmentation

However, the current system has a significant limitation: **when GPU memory fills up, blocks are offloaded to CPU memory only**. This presents several challenges:

```
Current Flow:
┌────────────┐     ┌────────────┐
│ GPU Memory │ ←→  │ CPU Memory │  ← Limited by DRAM capacity
│ (KV Cache) │     │ (Offload)  │  ← CPU involved in transfers
└────────────┘     └────────────┘
```

**Limitations of CPU-only offloading:**
1. **Limited capacity**: CPU RAM is finite and expensive to scale
2. **CPU overhead**: Data transfers involve CPU memory copies and bandwidth
3. **Cost**: Adding more RAM requires expensive hardware upgrades
4. **Scalability**: Each machine has fixed RAM capacity

### 1.3 Our Proposal: SSD Storage Tier with GPU Direct Storage

We propose adding an **SSD storage tier** for KV cache blocks, leveraging **NVIDIA GPU Direct Storage (GDS)** for efficient data movement:

```
Our Solution:
┌────────────┐     ┌────────────┐     ┌────────────┐
│ GPU Memory │ ←→  │ CPU Memory │ ←→  │  NVMe SSD  │
│ (KV Cache) │     │ (Optional) │     │ (Extended  │
└────────────┘     └────────────┘     │   Cache)   │
       │                              └────────────┘
       │           GPU Direct Storage        ↑
       └────────────────────────────────────→┘
              (Bypass CPU entirely)
```

### 1.4 Why SSD? Economic and Technical Rationale

| Resource | Cost (per TB) | Capacity Scaling | Hardware Changes |
|----------|---------------|------------------|------------------|
| GPU HBM | ~$10,000+ | Very Limited | New GPU required |
| CPU RAM | ~$100-200 | Limited (slots) | Motherboard constraints |
| NVMe SSD | ~$50-100 | Virtually unlimited | Hot-swappable |

**Key benefits of SSD offloading:**

1. **Cost efficiency**: SSDs provide 10-100x more storage per dollar than GPU/CPU memory
2. **Capacity**: Expand to terabytes without hardware reconfiguration
3. **No GPU changes**: Maximize existing GPU investment
4. **GDS advantage**: Direct GPU-to-SSD transfers bypass CPU, reducing latency

### 1.5 Why PagedAttention is Ideal for SSD Integration

PagedAttention's block-based architecture naturally aligns with SSD I/O characteristics:

- **Block granularity**: KV blocks map directly to SSD block I/O operations
- **Sequential access patterns**: Prefill and decoding access blocks sequentially
- **Reuse tracking**: Block hashes enable efficient cache lookups
- **Eviction policies**: LRU/ARC policies translate directly to SSD cache management

---

## 2. Overview of Our Solution

### 2.1 Architecture Overview

Our implementation introduces a complete SSD offloading subsystem within vLLM's v1 architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                      vLLM Scheduler                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           OffloadingConnector (Scheduler Side)           │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │            OffloadingManager                      │   │    │
│  │  │  • LRUOffloadingManager or ARCOffloadingManager  │   │    │
│  │  │  • Tracks block locations (GPU vs SSD)           │   │    │
│  │  │  • Manages eviction policy                        │   │    │
│  │  │  • Coordinates load/store operations              │   │    │
│  │  └──────────────────────────────────────────────────┘   │    │
│  │                         │                                │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │              SSDBackend                           │   │    │
│  │  │  • Block allocation on SSD                        │   │    │
│  │  │  • File offset management                         │   │    │
│  │  │  • Free list tracking                             │   │    │
│  │  └──────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                    KVConnectorMetadata
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        vLLM Worker                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           OffloadingConnector (Worker Side)              │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │            OffloadingWorker                       │   │    │
│  │  │  • Routes transfers to appropriate handlers       │   │    │
│  │  │  • Tracks pending transfer jobs                   │   │    │
│  │  └──────────────────────────────────────────────────┘   │    │
│  │                         │                                │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │         GpuSsdOffloadingHandler                   │   │    │
│  │  │  • Uses kvikio/cuFile for GDS transfers          │   │    │
│  │  │  • Async I/O with ThreadPoolExecutor             │   │    │
│  │  │  • Manages pre-allocated cache files             │   │    │
│  │  └──────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    ┌───────────────┐                   ┌───────────────┐
    │   GPU Memory  │  ◄── GDS ──►      │   NVMe SSD    │
    │   (KV Cache)  │                   │ (Cache Files) │
    └───────────────┘                   └───────────────┘
```

### 2.2 Key Components

#### 2.2.1 Abstract Interfaces (`abstract.py`, `spec.py`, `backend.py`)

We designed a clean abstraction layer that enables pluggable backends:

```python
class OffloadingManager(ABC):
    """Scheduler-side manager for tracking offloaded blocks"""
    
    def lookup(self, block_hashes) -> int:
        """Find consecutive hits in offload cache"""
    
    def prepare_load(self, block_hashes) -> LoadStoreSpec:
        """Prepare blocks for loading, protect from eviction"""
    
    def prepare_store(self, block_hashes) -> PrepareStoreOutput:
        """Prepare blocks for storing, handle evictions"""
    
    def complete_load/complete_store(self, block_hashes):
        """Mark transfers as complete"""
```

```python
class Backend(ABC):
    """Abstract storage backend (CPU, SSD, etc.)"""
    
    def allocate_blocks(self, block_hashes) -> list[BlockStatus]
    def free(self, block: BlockStatus)
    def get_num_free_blocks() -> int
```

#### 2.2.2 LRU and ARC Eviction Managers (`lru_manager.py`, `arc_manager.py`)

We implemented two eviction policies:

**LRUOffloadingManager**: Simple Least Recently Used eviction
- Maintains ordered dictionary of blocks
- O(1) lookup, insertion, and eviction
- Best for workloads with clear temporal locality

**ARCOffloadingManager**: Adaptive Replacement Cache
- Maintains T1 (recent) and T2 (frequent) caches
- Ghost lists B1/B2 for adaptive learning
- Self-tunes between recency and frequency
- Better for mixed access patterns

```python
class ARCOffloadingManager(OffloadingManager):
    """
    ARC Algorithm Flow:
    1. New blocks enter T1 (recent)
    2. Second access promotes to T2 (frequent)
    3. Eviction based on adaptive target_t1_size
    4. Ghost lists adjust target based on miss patterns
    """
```

#### 2.2.3 SSD Backend (`backends/ssd.py`)

The SSD backend manages block allocation on NVMe storage:

```python
class SSDBackend(Backend):
    """Manages KV cache blocks on SSD storage"""
    
    def __init__(self, block_size, num_blocks, ssd_cache_dir, cache_file_prefix):
        # Pre-allocate cache files for each layer
        # Maintain free list for block reuse
    
    def allocate_blocks(self, block_hashes):
        # Allocate new blocks or reuse freed ones
        # Return SSDBlockStatus with file offsets
    
    def get_cache_file_path(self, layer_idx) -> Path:
        # Returns: /cache_dir/kv_cache_layer_{idx}.bin
```

#### 2.2.4 GPU-SSD Transfer Handler (`worker/gpu_ssd.py`)

The worker-side handler performs actual data transfers using kvikio:

```python
class GpuSsdOffloadingHandler(OffloadingHandler):
    """Handler for GPU-SSD transfers using GPU Direct Storage"""
    
    def __init__(self, ...):
        # Initialize kvikio
        # Create ThreadPoolExecutor for async I/O
        # Pre-allocate cache files with posix_fallocate
    
    def transfer_async(self, job_id, spec) -> bool:
        # Submit transfer to thread pool
        # Track pending transfers by job_id
    
    def _do_gpu_to_ssd_transfer(self, src_blocks, dst_blocks):
        # For each layer:
        with kvikio.CuFile(file_path, "r+") as ssd_file:
            for src_block, dst_block in zip(...):
                # Copy GPU block to contiguous buffer
                # Write to SSD using GDS pwrite()
    
    def _do_ssd_to_gpu_transfer(self, src_blocks, dst_blocks):
        # For each layer:
        with kvikio.CuFile(file_path, "r") as ssd_file:
            for src_block, dst_block in zip(...):
                # Read from SSD using GDS pread()
                # Copy to GPU cache using cudaMemcpy
```

#### 2.2.5 kvikio Integration for GPU Direct Storage

We leverage the **kvikio** library from RAPIDS for GPU Direct Storage:

```python
# Check GDS availability
import kvikio
props = kvikio.DriverProperties()
if props.is_gds_available:
    # Direct GPU-NVMe transfers enabled
    # Major/minor version: props.major_version, props.minor_version

# File operations
with kvikio.CuFile(path, "r+") as f:
    # Async write: GPU buffer → SSD
    future = f.pwrite(gpu_tensor, file_offset=offset)
    nbytes = future.get()  # Wait for completion
    
    # Async read: SSD → GPU buffer
    future = f.pread(gpu_buffer, file_offset=offset)
    nbytes = future.get()
```

**Benefits of kvikio/GDS:**
- Direct DMA transfers between GPU and NVMe
- Bypasses CPU memory and page cache
- Asynchronous I/O with futures
- Automatic fallback to compatibility mode if GDS unavailable

#### 2.2.6 Connector Architecture (`offloading_connector.py`)

The OffloadingConnector bridges scheduler and worker:

```python
class OffloadingConnector(KVConnectorBase_V1):
    """Main entry point for offloading integration"""
    
    # Scheduler-side methods:
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """Check how many tokens can be loaded from SSD cache"""
    
    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        """Track blocks that need loading"""
    
    def build_connector_meta(self, scheduler_output):
        """Build metadata with load/store requests for workers"""
    
    # Worker-side methods:
    def start_load_kv(self, forward_context):
        """Initiate async loading of KV data"""
    
    def wait_for_save(self):
        """Initiate async saving of new KV data"""
    
    def get_finished(self, finished_req_ids):
        """Get completed transfer requests"""
```

### 2.3 Data Flow

#### Store (GPU → SSD):
```
1. Request generates new KV cache blocks
2. Scheduler identifies blocks to offload (excess blocks)
3. OffloadingManager.prepare_store() allocates SSD blocks, evicts if needed
4. KVConnectorMetadata carries store specs to worker
5. GpuSsdOffloadingHandler.transfer_async() submits write job
6. Worker thread: kvikio.CuFile.pwrite() transfers GPU→SSD
7. Worker reports completion to scheduler
8. OffloadingManager.complete_store() marks blocks as ready
```

#### Load (SSD → GPU):
```
1. New request needs previously computed KV cache
2. Scheduler checks OffloadingManager.lookup() for hits
3. OffloadingManager.prepare_load() protects blocks from eviction
4. KVConnectorMetadata carries load specs to worker
5. GpuSsdOffloadingHandler.transfer_async() submits read job
6. Worker thread: kvikio.CuFile.pread() transfers SSD→GPU
7. Request scheduled after transfer completion
8. OffloadingManager.complete_load() releases eviction protection
```

### 2.4 Design Decisions and Rationale

#### Why this structure?

1. **Separation of concerns**: Scheduler tracks metadata (block locations, eviction), worker handles I/O
2. **Pluggable backends**: Same manager code works with CPU, SSD, or future backends
3. **Async I/O**: Non-blocking transfers maximize GPU utilization
4. **Block size flexibility**: SSD blocks can be multiples of GPU blocks for I/O efficiency

#### Async Architecture Benefits:
- GPU can continue serving other requests during I/O
- Multiple transfers can be in flight concurrently
- Thread pool prevents blocking the main worker loop

### 2.5 Challenges Encountered

#### Challenge 1: PyTorch Inference Mode Restrictions
**Problem**: PyTorch's inference mode prevents in-place tensor modifications
**Solution**: Use direct CUDA memcpy via ctypes to bypass restrictions

```python
def _cuda_memcpy(dst_ptr, src_ptr, size):
    """Direct CUDA memcpy to bypass inference mode"""
    cudart.cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice)
```

#### Challenge 2: Variable KV Cache Layouts
**Problem**: Different attention backends have different tensor layouts
**Solution**: Detect layout by probing backend's get_kv_cache_shape()

```python
test_shape = attn_backend.get_kv_cache_shape(num_blocks=1234, ...)
if test_shape[0] == 2:  # [2, num_blocks, ...]
    kv_dim_before_num_blocks = True
else:  # [num_blocks, ...]
    kv_dim_before_num_blocks = False
```

#### Challenge 3: Block Size Alignment
**Problem**: SSD I/O is more efficient with larger blocks
**Solution**: Support configurable SSD block size as multiple of GPU block size

```python
ssd_block_size = gpu_block_size * block_size_factor
# E.g., 4 GPU blocks (16 tokens each) = 1 SSD block (64 tokens)
```

#### Challenge 4: Reference Counting During Transfers
**Problem**: Blocks being transferred cannot be evicted
**Solution**: ref_cnt tracking with -1 for "not ready" state

```python
class BlockStatus:
    ref_cnt: int  # -1 = not ready, 0 = ready, >0 = in use
    
    @property
    def is_ready(self) -> bool:
        return self.ref_cnt >= 0
```

### 2.6 Alternatives Considered

#### Alternative 1: Direct GPU-to-SSD without kvikio
**Considered**: Implementing GDS integration from scratch
**Discarded**: kvikio is battle-tested, maintained by NVIDIA/RAPIDS, handles edge cases

#### Alternative 2: Synchronous I/O
**Considered**: Blocking transfers in the main worker loop
**Discarded**: Would stall GPU and reduce throughput significantly

#### Alternative 3: Memory-mapped files
**Considered**: mmap() with GPU access
**Discarded**: Doesn't leverage GDS, still requires CPU page cache

#### Alternative 4: Custom eviction policy
**Considered**: Request-aware eviction (evict oldest request's blocks)
**Discarded**: LRU/ARC are proven, request-aware adds complexity without clear benefit

---

## 3. Implementation Details

### 3.1 File Structure

```
vllm/v1/kv_offload/
├── __init__.py
├── abstract.py          # OffloadingManager, LoadStoreSpec interfaces
├── spec.py              # OffloadingSpec base class
├── factory.py           # Spec factory with lazy loading
├── backend.py           # Backend, BlockStatus interfaces
├── lru_manager.py       # LRU eviction policy manager
├── arc_manager.py       # ARC eviction policy manager
├── mediums.py           # GPULoadStoreSpec, CPULoadStoreSpec, SSDLoadStoreSpec
├── cpu.py               # CPUOffloadingSpec
├── ssd.py               # SSDOffloadingSpec
├── backends/
│   ├── cpu.py           # CPUBackend
│   └── ssd.py           # SSDBackend
└── worker/
    ├── worker.py        # OffloadingHandler, OffloadingWorker
    ├── cpu_gpu.py       # CpuGpuOffloadingHandler
    └── gpu_ssd.py       # GpuSsdOffloadingHandler (kvikio integration)

vllm/distributed/kv_transfer/kv_connector/v1/
└── offloading_connector.py  # OffloadingConnector, scheduler/worker integration
```

### 3.2 Configuration

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "spec_name": "SSDOffloadingSpec",
            "num_ssd_blocks": 10000,
            "ssd_cache_dir": "/mnt/nvme/vllm_cache",
            "eviction_policy": "lru",
            "num_io_threads": 4,
            "block_size": 64
        }
    }'
```

### 3.3 Requirements

**Hardware:**
- NVIDIA GPU (Volta or newer, compute capability 7.0+)
- NVMe SSD with GDS-compatible filesystem (ext4, XFS)

**Software:**
- CUDA 11.4+ with GDS support
- kvikio library: `pip install kvikio-cu12`
- nvidia-gds package for GDS drivers

### 3.4 Testing

We implemented comprehensive tests:

```
tests/v1/kv_offload/
├── test_ssd_manager.py       # Unit tests for SSDBackend, LRU/ARC managers
├── test_gpu_ssd.py           # GPU-SSD transfer handler tests
├── test_ssd_integration.py   # End-to-end integration tests
└── test_cpu_manager.py       # CPU backend tests (for comparison)
```

**Test coverage includes:**
- Backend initialization and block allocation
- LRU/ARC eviction behavior
- GPU↔SSD transfer correctness
- Data integrity (roundtrip verification)
- Concurrent transfers
- Event emission for observability

---

## 4. Comparison: CPU vs SSD Offloading

| Aspect | CPU Offloading | SSD Offloading (GDS) |
|--------|---------------|---------------------|
| **Latency** | ~10-20 μs | ~100-500 μs |
| **Throughput** | ~20-50 GB/s (DDR5) | ~5-15 GB/s (NVMe Gen4) |
| **Capacity** | 100s GB (limited by DIMM slots) | TBs (virtually unlimited) |
| **CPU Usage** | High (memory copies) | Minimal (GDS bypass) |
| **Cost per GB** | ~$5-10/GB | ~$0.05-0.10/GB |
| **Best for** | Latency-sensitive, small cache | Large cache, long context |

---

## 5. Future Work

1. **Tiered caching**: Combine CPU and SSD tiers (CPU as L2, SSD as L3)
2. **Predictive prefetching**: Anticipate which blocks will be needed
3. **Compression**: Compress KV data before SSD storage
4. **RDMA support**: Extend to network-attached storage
5. **Multi-GPU coordination**: Shared SSD cache across GPUs

---

## 6. Conclusion

We have successfully implemented an SSD-based KV cache offloading system for vLLM's PagedAttention architecture. Key achievements:

1. **Modular design**: Clean separation between managers, backends, and handlers
2. **GDS integration**: Direct GPU-SSD transfers via kvikio
3. **Multiple eviction policies**: LRU and ARC support
4. **Async I/O**: Non-blocking transfers for maximum throughput
5. **Comprehensive testing**: Unit, integration, and data integrity tests

This implementation enables vLLM to serve larger effective batch sizes and longer context lengths by extending the KV cache to cost-effective SSD storage, without requiring expensive GPU or CPU memory upgrades.

---

## References

1. [vLLM: Easy, Fast, and Cheap LLM Serving](https://vllm.ai/)
2. [PagedAttention: Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180)
3. [NVIDIA GPU Direct Storage](https://developer.nvidia.com/gpudirect-storage)
4. [kvikio: High-Performance File I/O for GPUs](https://github.com/rapidsai/kvikio)
5. [ARC: A Self-Tuning, Low Overhead Replacement Cache](https://www.usenix.org/conference/fast-03/arc-self-tuning-low-overhead-replacement-cache)
